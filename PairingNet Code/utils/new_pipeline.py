import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Start: Copied/Adapted from the new RCRI_CM.py ---
# (Functions are mostly identical to _LRA version, but compute_LOA default DIR=True is key)

def downsampling(num_sample, contour, LOA=None):
    B, N, C = contour.shape
    contour = contour.contiguous()

    if N == 0:  # Handle empty contour before min
        return contour.new_empty(B, 0, C), None, contour.new_empty(B, 0).long()

    # Ensure num_sample is not greater than N
    # If num_sample is None (for global_cm=True), num_index should be N
    if num_sample is not None:
        num_index = min(N, num_sample)
    else:
        num_index = N

    if num_index == 0:  # If N was >0 but num_sample was 0 (unlikely but safe)
        return contour.new_empty(B, 0, C), None, contour.new_empty(B, 0).long()

    new_idx_float = torch.linspace(0, N - 1, num_index, device=contour.device)
    new_index = new_idx_float.long()

    sampled_points = contour[:, new_index, :]
    if LOA is not None:
        new_LOA = LOA[:, new_index, :]
    else:
        new_LOA = None
    # Ensure new_index is 2D for expand for B=0 case
    new_index_expanded = new_index.unsqueeze(0).expand(B, -1) if B > 0 else new_index.view(1, -1).expand(B, -1)

    return sampled_points, new_LOA, new_index_expanded


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    if B == 0:  # Handle empty batch
        return points.new_empty(0, idx.shape[1] if idx.ndim > 1 else 0, points.shape[-1])

    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def index_in_LGA(num_neighbors, total_num_points_in_contour, new_interest_point_indices):
    B, S = new_interest_point_indices.shape

    if total_num_points_in_contour == 0:
        return new_interest_point_indices.new_empty(B, S, 0).long()

    actual_num_neighbors = min(num_neighbors, total_num_points_in_contour)
    if actual_num_neighbors == 0:
        return new_interest_point_indices.new_empty(B, S, 0).long()

    new_index_expanded = new_interest_point_indices.unsqueeze(2)

    half_neighbors_floor = actual_num_neighbors // 2

    start_id = new_index_expanded - half_neighbors_floor

    range_idx = torch.arange(0, actual_num_neighbors, device=new_interest_point_indices.device).view(1, 1, -1).expand(B,
                                                                                                                      S,
                                                                                                                      -1)

    group_idx = (start_id + range_idx) % total_num_points_in_contour
    return group_idx


def adjust_normals(P, N):  # From RCRI_CM.py
    K = 32  # K_dir in my previous code for centroid of Nhi
    num_batch, num_p, C_dim = P.shape
    if num_p == 0: return N.clone()

    adjusted_N = N.clone()

    # Ensure K is not larger than num_p
    actual_K_for_centroid = min(K, num_p)
    if actual_K_for_centroid <= 1 and num_p > 1:  # Need at least 2 points for a meaningful direction if K=1
        actual_K_for_centroid = min(2, num_p)
    elif actual_K_for_centroid == 0:  # cannot compute centroid
        return adjusted_N

    # centroid of K points "in front" (Nhi in paper)
    # indices are 0 to num_p-1
    indices = torch.arange(num_p, device=P.device).unsqueeze(0).expand(num_batch, -1)  # (num_batch, num_p)

    # Create indices for K points starting from the *next* point, circularly
    # The paper uses "Nhi located in front of ci"
    # So if current point is i, Nhi could be i+1, i+2, ...
    # K in adjust_normals in RCRI_CM.py seems to be for points *around* ci not strictly in front
    # "selected_points = P[torch.arange(P.shape[0])... indices_K]" uses indices_K = (indices + range(K)) % num_p
    # This means it takes K points *starting from* ci. For gi - ci, we need points offset from ci.
    # Let's follow the paper's Fig 3 & Eq 3 more closely: gi is centroid of Nhi (points *in front* of ci)

    # For gi (centroid of points in front):
    # Let's use a small number of points in front, e.g., K_front
    K_front = min(actual_K_for_centroid, num_p - 1 if num_p > 1 else 0)  # Ensure K_front < num_p
    if K_front == 0 and num_p > 1: K_front = 1  # at least one point if possible
    if K_front == 0: return adjusted_N  # cannot determine direction

    nhi_indices_range = torch.arange(1, K_front + 1, device=P.device).view(1, 1, -1)  # 1,1,K_front
    nhi_indices = (indices.unsqueeze(2) + nhi_indices_range) % num_p  # B, num_p, K_front

    Nhi_points = index_points(P, nhi_indices)  # B, num_p, K_front, C
    gi = torch.mean(Nhi_points, dim=2)  # B, num_p, C (centroid of points in front)

    line_vecs = gi - P  # (num_batch, num_p, C) (vector gi - ci)

    # Cross product for 2D (components 0 and 1 if C_dim >= 2)
    # N is (nx, ny, ...), line_vecs is (dx, dy, ...)
    # N x line_vecs --> z-component is nx*dy - ny*dx
    if C_dim >= 2:
        cross_products_z = (N[:, :, 0] * line_vecs[:, :, 1]) - (N[:, :, 1] * line_vecs[:, :, 0])  # (num_batch, num_p)
        # Update Normal Vector: sgn(N x (gi-ci)) * N. If cross_product_z < 0, flip N.
        # The paper has LOAi = sgn(...) * ni. If sgn is -1, ni flips.
        # Original RCRI_CM adjust_normals: adjusted_N[cross_products > 0] = -N[cross_products > 0]
        # This means if (N x (gi-ci))_z > 0, flip N. This seems opposite to standard right-hand rule
        # if N is "outward" and contour is CCW. Let's stick to their implementation.
        # sgn_factor = torch.sign(cross_products_z).unsqueeze(-1)
        # sgn_factor[sgn_factor == 0] = 1
        # adjusted_N = sgn_factor * N # This would be LOAi = sgn * ni

        # Following RCRI_CM.py's adjust_normals:
        adjusted_N[cross_products_z > 0] = -N[cross_products_z > 0]

    return adjusted_N


def compute_LOA_one(group_xyz_centered, weighting=False):  # group_xyz_centered is local LGA points
    B, S, N_local, C = group_xyz_centered.shape
    if N_local == 0: return group_xyz_centered.new_zeros(B, S, C)

    dists = torch.norm(group_xyz_centered, dim=-1, keepdim=True)

    if weighting:
        dists_max, _ = dists.max(dim=2, keepdim=True)
        weights = dists_max - dists  # Higher weight for closer points to interest point
        dists_sum = weights.sum(dim=2, keepdim=True)
        weights = weights / (dists_sum + 1e-8)  # Normalize weights
        weights[weights != weights] = 1.0 / N_local if N_local > 0 else 0.0
        M = torch.matmul(group_xyz_centered.transpose(3, 2), weights * group_xyz_centered)
    else:
        M = torch.matmul(group_xyz_centered.transpose(3, 2), group_xyz_centered)

    try:
        eigen_values, vec = torch.linalg.eigh(M)
    except torch._C._LinAlgError:
        return group_xyz_centered.new_zeros(B, S, C)

    LRA = vec[:, :, :, 0]  # Smallest eigenvector direction
    LRA_length = torch.norm(LRA, dim=-1, keepdim=True)
    LRA = LRA / (LRA_length + 1e-8)
    return LRA


def compute_LOA(xyz, weighting=True, num_neighbors_for_loa_calc=64, DIR=True):  # Default DIR=True
    # xyz: [B, N_orig, C]
    B_orig, N_orig, C_orig = xyz.shape
    if N_orig == 0: return xyz.new_zeros(B_orig, N_orig, C_orig)

    actual_num_neighbors = min(num_neighbors_for_loa_calc, N_orig)
    if actual_num_neighbors == 0: return xyz.new_zeros(B_orig, N_orig, C_orig)

    # Using sequential neighbors for LOA calculation as implied by paper's Fig.3 (N_ri)
    interest_points_indices = torch.arange(N_orig, device=xyz.device).view(1, -1).expand(B_orig, -1)  # B_orig, N_orig

    # Get indices of neighbors for each point
    idx_local_neighborhood = index_in_LGA(actual_num_neighbors, N_orig, interest_points_indices)
    # idx_local_neighborhood: [B_orig, N_orig, actual_num_neighbors]

    group_xyz_all_neighborhoods = index_points(xyz, idx_local_neighborhood)
    # [B_orig, N_orig (as S), actual_num_neighbors (as N_local), C_orig]

    group_xyz_centered = group_xyz_all_neighborhoods - xyz.unsqueeze(2)

    # Compute initial LRA (normal direction) for each of the N_orig points
    LRA = compute_LOA_one(group_xyz_centered, weighting=weighting)  # [B_orig, N_orig, C_orig]

    if DIR:  # Orient the LRA to get LOA
        LRA = adjust_normals(xyz, LRA)  # xyz provides context for orientation

    LRA_length = torch.norm(LRA, dim=-1, keepdim=True)
    LOA = LRA / (LRA_length + 1e-8)  # Normalized LOA
    return LOA


def LOAI_features(contour_all_points, LOA_all_points,
                  sampled_interest_points, LOA_for_interest_points,
                  idx_for_LGA_points, global_cm=False):
    B, S, C = sampled_interest_points.shape
    num_LGA_neighbors_actual = idx_for_LGA_points.shape[-1]

    if num_LGA_neighbors_actual == 0:
        return sampled_interest_points.new_zeros(B, S, 0, 8), idx_for_LGA_points

    LOA_ci = LOA_for_interest_points.unsqueeze(2)  # [B, S, 1, C]
    epsilon = 1e-7

    all_LGA_points_xi = index_points(contour_all_points, idx_for_LGA_points)  # [B, S, LGA_N, C]
    LOA_xi = index_points(LOA_all_points, idx_for_LGA_points)  # [B, S, LGA_N, C]

    if not global_cm:
        vec_ci_to_xi = all_LGA_points_xi - sampled_interest_points.unsqueeze(2)
    else:  # For global_cm, sampled_interest_points is centroid, all_LGA_points_xi are already local
        # if LOAI_features is called with contour_local_to_centroid and zero_vector for sampled_interest_points
        vec_ci_to_xi = all_LGA_points_xi

        # d1 in paper, f4 in my prev code: ||xi - ci||
    f4_dist_ci_xi = torch.norm(vec_ci_to_xi, dim=-1, keepdim=True)
    unit_vec_ci_to_xi = vec_ci_to_xi / (f4_dist_ci_xi + epsilon)

    # f1_cos_angle_LOAci_vec_ci_xi (a1 in paper is angle, here cosine)
    a1_like_cos = (LOA_ci * unit_vec_ci_to_xi).sum(dim=-1, keepdim=True)

    # f2_cos_angle_LOAxi_vec_cixi (a2 in paper)
    a2_like_cos = (LOA_xi * (-unit_vec_ci_to_xi)).sum(dim=-1, keepdim=True)

    # f3_cos_angle_LOAci_LOAxi (a3 in paper's RCRI_CM.py uses acos(clamp) and D_0 sign)
    # Paper Eq5: f3 = cos(angle(LOAci, LOAxi))
    # RCRI_CM.py LOAI_features:
    # a3_val = torch.matmul(d1_norm/*LOA_xi*/, new_LOA/*LOA_ci*/) -> [B,S,LGA_N,1] if LOA_ci was [B,S,1,C]
    # This seems to be a direct dot product, so it's already cos(angle)
    # a3 = torch.acos(torch.clamp(a3_val, -1 + epsilon, 1-epsilon))
    # D_0 = (a1 < a2) ; a3 = D_0.float() * a3
    # This D_0 part seems to be an additional orientation/disambiguation step not in Eq5.
    # For simplicity and directness from Eq5, let's use the cosine.
    # If strict adherence to RCRI_CM.py's LOAI_features is needed, that logic must be copied.
    # Let's use the RCRI_CM.py's "a3" logic for f3.
    a3_val_dotproduct = (LOA_xi * LOA_ci).sum(dim=-1, keepdim=True)
    f3_angle_LOAci_LOAxi = torch.acos(torch.clamp(a3_val_dotproduct, -1 + epsilon, 1 - epsilon))
    D_0_condition = (a1_like_cos < a2_like_cos)  # Comparing cosines
    # 修改第242行
    D_0_sign = D_0_condition.float() * 2 - 1
    D_0_sign[a1_like_cos == a2_like_cos] = -1  # As per D_0[D_0 ==0] = -1
    f3_oriented_angle = D_0_sign * f3_angle_LOAci_LOAxi

    # Features between adjacent points xi and xi-1
    vec_xim1_to_xi = all_LGA_points_xi - torch.roll(all_LGA_points_xi, shifts=1, dims=2)
    if num_LGA_neighbors_actual > 0: vec_xim1_to_xi[:, :, 0, :] = 0  # Boundary for roll

    # f8_dist_xim1_xi (d_inner in paper)
    f8_dist_xim1_xi = torch.norm(vec_xim1_to_xi, dim=-1, keepdim=True)
    unit_vec_xim1_to_xi = vec_xim1_to_xi / (f8_dist_xim1_xi + epsilon)

    LOA_xim1 = torch.roll(LOA_xi, shifts=1, dims=2)
    if num_LGA_neighbors_actual > 0: LOA_xim1[:, :, 0, :] = 0

    # a4 in paper
    a4_like_cos = (unit_vec_xim1_to_xi * LOA_xi).sum(dim=-1, keepdim=True)  # Angle(xi-xim1, LOA_xi)

    # a5 in paper
    a5_like_cos = (unit_vec_xim1_to_xi * LOA_xim1).sum(dim=-1, keepdim=True)  # Angle(xi-xim1, LOA_xim1)
    # Note: paper RCRI_CM.py has a4 = (LGA_inner_unit * d1_norm/*LOA_xi*/).sum()
    #                          a5 = (LGA_inner_unit * torch.roll(d1_norm,1,2)/*LOA_xim1*/).sum()
    # This matches my a4_like_cos and a5_like_cos.

    # a6 in paper (angle(LOA_xi, LOA_xim1), with D_1 sign)
    # RCRI_CM.py LOAI_features:
    # a6_val = (d1_norm * torch.roll(d1_norm,1,2)).sum()
    # a6 = torch.acos(clamp(a6_val))
    # D_1 = (a4 < a5) ; a6 = D_1.float() * a6
    a6_val_dotproduct = (LOA_xi * LOA_xim1).sum(dim=-1, keepdim=True)
    f7_angle_LOAxi_LOAxim1 = torch.acos(torch.clamp(a6_val_dotproduct, -1 + epsilon, 1 - epsilon))
    D_1_condition = (a4_like_cos < a5_like_cos)
    # 修改第283行
    D_1_sign = D_1_condition.float() * 2 - 1
    D_1_sign[a4_like_cos == a5_like_cos] = -1
    f7_oriented_angle = D_1_sign * f7_angle_LOAxi_LOAxim1

    # d2 in paper (angle between (xi-ci) and (xi-1 - ci))
    # inner_angle_feat = (d1_unit * torch.roll(d1_unit,1,2)).sum()
    # d2 = acos(clamp(inner_angle_feat))
    # d1_unit is unit_vec_ci_to_xi
    rolled_unit_vec_ci_to_xi = torch.roll(unit_vec_ci_to_xi, shifts=1, dims=2)
    if num_LGA_neighbors_actual > 0: rolled_unit_vec_ci_to_xi[:, :, 0, :] = 0  # Boundary

    d2_dotproduct = (unit_vec_ci_to_xi * rolled_unit_vec_ci_to_xi).sum(dim=-1, keepdim=True)
    d2_angle_adj_vecs_from_ci = torch.acos(torch.clamp(d2_dotproduct, -1 + epsilon, 1 - epsilon))

    # Order from RCRI_CM.py: [d1, d2, a1, a2, a3, a4, a5, a6]
    # d1 -> f4_dist_ci_xi
    # d2 -> d2_angle_adj_vecs_from_ci
    # a1 -> a1_like_cos (cos of angle)
    # a2 -> a2_like_cos (cos of angle)
    # a3 -> f3_oriented_angle (actual angle with sign)
    # a4 -> a4_like_cos (cos of angle)
    # a5 -> a5_like_cos (cos of angle)
    # a6 -> f7_oriented_angle (actual angle with sign)

    RIF_LOA = torch.cat([
        f4_dist_ci_xi, d2_angle_adj_vecs_from_ci,
        a1_like_cos, a2_like_cos, f3_oriented_angle,
        a4_like_cos, a5_like_cos, f7_oriented_angle
    ], dim=-1)

    return RIF_LOA, idx_for_LGA_points


def Feature_Encoding(num_sample_interest_points, num_LGA_neighbors,
                     contour_all_points, LOA_all_points, global_cm):
    B, N_total, C = contour_all_points.shape

    if global_cm:
        # For global_cm, num_sample_interest_points and num_LGA_neighbors are often None in rcri_cm call
        # It means S=1 (one interest point, the centroid) and LGA_N = N_total (all points form LGA)
        sampled_interest_points_coords = torch.mean(contour_all_points, dim=1, keepdim=True)  # [B, 1, C]

        contour_local_to_centroid = contour_all_points - sampled_interest_points_coords

        # LOA for the single "interest point" (centroid)
        # compute_LOA_one needs [B, S, N_local_pts_for_LOA_calc, C]
        # Here, the LOA is for the centroid itself, derived from all points.
        LOA_for_interest_points = compute_LOA_one(contour_local_to_centroid.unsqueeze(1), weighting=True)  # [B, 1, C]

        idx_for_LGA_points = torch.arange(N_total, device=contour_all_points.device).long()
        idx_for_LGA_points = idx_for_LGA_points.view(1, 1, N_total).expand(B, 1, -1)

        # For LOAI_features in global_cm case:
        # contour argument becomes contour_local_to_centroid
        # sampled_points argument becomes effectively zero (as points are already local)
        # new_LOA argument is LOA_for_interest_points (LOA of centroid)
        RIF_LOA, idx_ordered = LOAI_features(
            contour_local_to_centroid, LOA_all_points,  # LOA_all_points are for original contour points
            torch.zeros_like(sampled_interest_points_coords),  # Points are local, so interest point is origin
            LOA_for_interest_points,
            idx_for_LGA_points, global_cm=True
        )
        # sampled_points return is None as per original Feature_Encoding for global_cm
        return None, RIF_LOA, LOA_for_interest_points, idx_ordered

    else:  # Standard fragment/LGA encoding
        sampled_interest_points_coords, LOA_for_interest_points, new_interest_point_indices = \
            downsampling(num_sample_interest_points, contour_all_points, LOA_all_points)

        idx_for_LGA_points = index_in_LGA(num_LGA_neighbors, N_total, new_interest_point_indices)

        RIF_LOA, idx_ordered = LOAI_features(
            contour_all_points, LOA_all_points,
            sampled_interest_points_coords, LOA_for_interest_points,
            idx_for_LGA_points, global_cm=False
        )
        return sampled_interest_points_coords, RIF_LOA, LOA_for_interest_points, idx_ordered


class rcri_cm_adapted_non_lra(nn.Module):  # Renamed to reflect source
    def __init__(self, num_sample_interest_points, num_LGA_neighbors,
                 filter_in_channel_prev_feature,
                 filter_output_channels_list,
                 global_cm_flag):
        super(rcri_cm_adapted_non_lra, self).__init__()
        self.num_sample_interest_points = num_sample_interest_points
        self.num_LGA_neighbors = num_LGA_neighbors
        self.global_cm_flag = global_cm_flag

        self.feature_enhancement_convs = nn.ModuleList()
        self.feature_enhancement_bns = nn.ModuleList()
        RIF_channel_count = 8
        current_enhancement_channel = RIF_channel_count
        feature_enhancement_output_channels = [32, 64]

        for out_ch in feature_enhancement_output_channels:
            self.feature_enhancement_convs.append(nn.Conv2d(current_enhancement_channel, out_ch, 1))
            self.feature_enhancement_bns.append(nn.BatchNorm2d(out_ch))
            current_enhancement_channel = out_ch

        self.final_enhancement_dim = current_enhancement_channel

        self.filter_convs = nn.ModuleList()
        self.filter_bns = nn.ModuleList()

        # Input to filter is (enhanced RIF features) + (features from previous layer if concatenated)
        # Original RCRI_CM.py: temp_channel = filter_in_channel (which is prev_feat_dim + enhanced_RIF_dim)
        # My previous code was: current_filter_channel = self.final_enhancement_dim + filter_in_channel_prev_feature
        # The `filter_in_channel` arg to original `rcri_cm` is `prev_output_dim + channel_input (64 for enhanced RIF)`
        # So, filter_in_channel_prev_feature corresponds to prev_output_dim.
        # Thus, actual input channel to first filter_conv is self.final_enhancement_dim + filter_in_channel_prev_feature
        current_filter_channel = self.final_enhancement_dim + filter_in_channel_prev_feature

        for out_ch in filter_output_channels_list:
            self.filter_convs.append(nn.Conv2d(current_filter_channel, out_ch, 1))
            self.filter_bns.append(nn.BatchNorm2d(out_ch))
            current_filter_channel = out_ch

        self.final_output_dim = current_filter_channel

    def forward(self, contour_all_points, LOA_all_points, feature_maps_from_prev_layer):
        # contour_all_points: [B, N_total, C]
        # LOA_all_points: [B, N_total, C]
        # feature_maps_from_prev_layer: [B, prev_feature_dim, S_prev_samples] (channels first) or None

        B_batch, N_total_pts, C_dim = contour_all_points.shape

        # Feature Encoding based on LOAI (from RCRI_CM.py Feature_Encoding)
        # Note: num_sample for Feature_Encoding is self.num_sample_interest_points
        sampled_interest_points_coords, RIF_feature_map_raw, LOA_for_interest_points, idx_ordered_LGA_points = \
            Feature_Encoding(self.num_sample_interest_points, self.num_LGA_neighbors,
                             contour_all_points, LOA_all_points, self.global_cm_flag)
        # RIF_feature_map_raw: [B, S_samples, num_LGA_neigh, 8]

        # Permute RIF for Conv2d: [B, 8, num_LGA_neigh, S_samples]
        current_features_RIF = RIF_feature_map_raw.permute(0, 3, 2, 1)

        # Feature Enhancement (on the 8 RIFs)
        for i, conv_layer in enumerate(self.feature_enhancement_convs):
            bn_layer = self.feature_enhancement_bns[i]
            current_features_RIF = F.relu(bn_layer(conv_layer(current_features_RIF)))
        # current_features_RIF: [B, self.final_enhancement_dim (64), num_LGA_neigh, S_samples]

        # Concatenate with features from previous layer (if any)
        # This follows structure of RCRI_CM.py's forward
        if feature_maps_from_prev_layer is not None:
            # feature_maps_from_prev_layer is [B, prev_feat_dim, S_prev] (channels first)
            # Needs to be permuted to [B, S_prev, prev_feat_dim] for index_points
            feature_maps_from_prev_layer_permuted = feature_maps_from_prev_layer.permute(0, 2, 1)

            if not self.global_cm_flag:
                # If not global, S_prev should be N_total (if prev layer output all points)
                # or S_prev are the interest points sampled by previous layer.
                # idx_ordered_LGA_points: [B, S_samples, num_LGA_neigh]
                # These indices are into the original contour_all_points.
                # If feature_maps_from_prev_layer has features for *each point* in contour_all_points:
                grouped_prev_features = index_points(feature_maps_from_prev_layer_permuted, idx_ordered_LGA_points)
                # grouped_prev_features: [B, S_samples, num_LGA_neigh, prev_feat_dim]
                grouped_prev_features = grouped_prev_features.permute(0, 3, 2, 1)
                # grouped_prev_features: [B, prev_feat_dim, num_LGA_neigh, S_samples]
            else:  # global_cm case
                # Feature_maps from prev layer would be [B, prev_feat_dim, S_prev_samples (N_total before global)]
                # For global, S_samples = 1. num_LGA_neigh = N_total.
                # Original RCRI_CM.py: grouped_points = feature_maps.view(B, 1, N, -1) then permute.
                # This means feature_maps is [B, prev_dim, N_total_from_prev_stage_output]
                # And it's reshaped to be [B, prev_dim, N_total, 1 (S_samples for global)]
                # For our standalone encoder, feature_maps_from_prev_layer is None.
                # If it was part of a chain, this logic needs care.
                # Let's assume for standalone, this branch is not hit if prev_features is None.
                # If prev_features is NOT None and global_cm=True, it implies prev_features are for the
                # N_total points that are now forming the single global LGA.
                # S_samples for global is 1. num_LGA_neigh is N_total.
                # So prev_features [B, prev_dim, N_total] needs to become [B, prev_dim, N_total, 1]
                num_prev_feat_dim = feature_maps_from_prev_layer.shape[1]
                grouped_prev_features = feature_maps_from_prev_layer.view(B_batch, num_prev_feat_dim, N_total_pts, 1)

            features_to_filter = torch.cat([current_features_RIF, grouped_prev_features], dim=1)
        else:
            features_to_filter = current_features_RIF
        # features_to_filter: [B, (self.final_enhancement_dim + prev_feat_dim), num_LGA_neigh, S_samples]

        # Fuse features (filter convs)
        for i, conv_layer in enumerate(self.filter_convs):
            bn_layer = self.filter_bns[i]
            features_to_filter = F.relu(bn_layer(conv_layer(features_to_filter)))
        # features_to_filter: [B, self.final_output_dim, num_LGA_neigh, S_samples]

        # Max Pooling over LGA_neighbors dimension
        output_feature_map_per_LGA = torch.max(features_to_filter, dim=2)[0]
        # output_feature_map_per_LGA: [B, self.final_output_dim, S_samples]

        # sampled_interest_points_coords, LOA_for_interest_points are from Feature_Encoding
        return sampled_interest_points_coords, LOA_for_interest_points, output_feature_map_per_LGA


# --- End: Copied/Adapted from RCRI_CM.py ---


class ContourFragmentEncoder(nn.Module):
    def __init__(self, args, num_fragments=None, output_dim=None):
        super(ContourFragmentEncoder, self).__init__()

        # Calculate/extract parameters from args or use provided values
        self.fragment_length = 16  # Default fragment length (LGA size)
        
        # Use explicitly provided parameters if given, otherwise extract from args
        if num_fragments is not None:
            self.num_fragments = num_fragments
        else:
            # Use args.max_length directly to ensure compatibility with expected output size
            self.num_fragments = args.max_length
            
        if output_dim is not None:
            self.output_dim = output_dim
        else:
            self.output_dim = args.feature_dim  # Use feature dimension from args
            
        # num_neighbors for LOA calculation, distinct from fragment_length (LGA size)
        self.loa_calc_neighbors = min(64, args.max_length // 8)  # Default value adapted to input size

        self.encoder = rcri_cm_adapted_non_lra(  # Use the one based on RCRI_CM.py
            num_sample_interest_points=self.num_fragments,
            num_LGA_neighbors=self.fragment_length,
            filter_in_channel_prev_feature=0,
            filter_output_channels_list=[self.output_dim],
            global_cm_flag=False
        )

    def forward(self, inputs: dict):
        pcd_2d = inputs['pcd']
        bs, num_nodes, _ = pcd_2d.shape

        if num_nodes == 0:
            l_c = pcd_2d.new_zeros(bs, self.num_fragments, self.output_dim)
            q = torch.cat((l_c, torch.zeros_like(l_c)), dim=-1)
            w_ = torch.ones(bs, self.num_fragments, 1, device=l_c.device)
            return l_c, q, w_

        pcd_3d = F.pad(pcd_2d, (0, 1), "constant", 0)

        # Compute LOA for all points using RCRI_CM.py's compute_LOA (DIR=True default)
        # The num_neighbors here is for the local PCA for normal estimation (m in paper Fig.3)
        loa_all_points_3d = compute_LOA(
            pcd_3d,
            weighting=True,
            num_neighbors_for_loa_calc=self.loa_calc_neighbors,  # Explicitly pass
            DIR=True  # Ensure oriented LOA
        )

        _sampled_coords, _sampled_loa, fragment_features = \
            self.encoder(pcd_3d, loa_all_points_3d, feature_maps_from_prev_layer=None)

        l_c = fragment_features.permute(0, 2, 1)

        l_t_dummy = torch.zeros_like(l_c)
        q = torch.cat((l_c, l_t_dummy), dim=-1)
        w_ = torch.ones(bs, self.num_fragments, 1, device=l_c.device)

        return l_c, q, w_


if __name__ == '__main__':
    bs = 2
    num_contour_points = 50

    # Create a mock args object with necessary attributes
    class Args:
        def __init__(self):
            self.max_length = 40  # This would create 10 fragments (max_length // 4)
            self.feature_dim = 64

    args = Args()

    # Create encoder using the args object
    encoder = ContourFragmentEncoder(args)

    angles = torch.linspace(0, 2 * torch.pi, num_contour_points, dtype=torch.float32).unsqueeze(0).expand(bs, -1)
    radius = 1.0 + 0.1 * torch.rand(bs, num_contour_points)
    x = radius * torch.cos(angles)
    y = radius * torch.sin(angles)
    dummy_pcd = torch.stack([x, y], dim=-1)

    inputs_dict = {'pcd': dummy_pcd}

    l_c, q, w_ = encoder(inputs_dict)
    print("l_c shape:", l_c.shape)
    print("q shape:", q.shape)
    print("w_ shape:", w_.shape)

    dummy_pcd_empty = torch.rand(bs, 0, 2)
    inputs_dict_empty = {'pcd': dummy_pcd_empty}
    l_c_e, q_e, w_e = encoder(inputs_dict_empty)
    print("\nFor empty PCD:")
    print("l_c_e shape:", l_c_e.shape)
    print("q_e shape:", q_e.shape)
    print("w_e shape:", w_e.shape)

    num_short_nodes = 5  # Test N < loa_calc_neighbors and N < fragment_length
    assert num_short_nodes < encoder.loa_calc_neighbors
    assert num_short_nodes < encoder.fragment_length
    dummy_pcd_short = torch.rand(bs, num_short_nodes, 2)
    inputs_dict_short = {'pcd': dummy_pcd_short}
    l_c_s, q_s, w_s = encoder(inputs_dict_short)
    print("\nFor short PCD (N < fragment_length and N < loa_calc_neighbors):")
    print("l_c_s shape:", l_c_s.shape)
    print("q_s shape:", q_s.shape)
    print("w_s shape:", w_s.shape)
