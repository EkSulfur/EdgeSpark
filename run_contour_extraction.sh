#!/bin/bash

# PairingNet Contour特征提取完整执行脚本
# 功能：训练模型、提取contour模块、运行测试，支持日志重定向和后台执行

# 配置变量
PROJECT_DIR="$HOME/PairingNet/PairingNet Code"
LOG_DIR="$PROJECT_DIR/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="$LOG_DIR/contour_extraction_$TIMESTAMP.log"
TRAIN_LOG="$LOG_DIR/train_stage1_$TIMESTAMP.log"
EXTRACT_LOG="$LOG_DIR/extract_module_$TIMESTAMP.log"
TEST_LOG="$LOG_DIR/test_contour_$TIMESTAMP.log"
PID_FILE="$LOG_DIR/contour_extraction.pid"

# 创建日志目录
mkdir -p "$LOG_DIR"

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MAIN_LOG"
}

# 错误处理函数
error_exit() {
    log "ERROR: $1"
    exit 1
}

# 检查依赖函数
check_dependencies() {
    log "检查依赖环境..."
    
    # 检查Python环境
    if ! command -v python &> /dev/null; then
        error_exit "Python未安装或未在PATH中"
    fi
    
    # 检查CUDA环境
    if ! python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
        log "WARNING: CUDA环境不可用，可能影响训练性能"
    fi
    
    # 检查必要的Python包
    python -c "import torch, pickle, numpy" 2>/dev/null || error_exit "缺少必要的Python包"
    
    log "依赖检查完成"
}

# 检查数据集函数
check_dataset() {
    log "检查数据集文件..."
    
    local dataset_dir="$HOME/PairingNet/dataset"
    local required_files=("train_set_with_downsample.pkl" "valid_set_with_downsample.pkl" "test_set_with_downsample.pkl")
    
    if [ ! -d "$dataset_dir" ]; then
        error_exit "数据集目录不存在: $dataset_dir"
    fi
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$dataset_dir/$file" ]; then
            error_exit "缺少数据集文件: $file"
        fi
    done
    
    log "数据集检查完成"
}

# 训练STAGE_ONE模型函数
train_stage1() {
    log "开始训练STAGE_ONE模型..."
    
    cd "$PROJECT_DIR" || error_exit "无法进入项目目录"
    
    # 检查训练脚本是否存在
    if [ ! -f "train_stage1.py" ]; then
        error_exit "训练脚本不存在: train_stage1.py"
    fi
    
    # 执行训练，输出到日志文件
    python train_stage1.py > "$TRAIN_LOG" 2>&1
    local train_exit_code=$?
    
    if [ $train_exit_code -ne 0 ]; then
        error_exit "训练失败，请查看日志: $TRAIN_LOG"
    fi
    
    # 检查模型检查点是否生成
    local checkpoint_dir="$PROJECT_DIR/EXP/stage1_contour_extraction/checkpoint"
    if [ ! -d "$checkpoint_dir" ] || [ -z "$(ls -A "$checkpoint_dir")" ]; then
        error_exit "训练完成但未找到模型检查点"
    fi
    
    log "STAGE_ONE模型训练完成"
}

# 提取contour特征模块函数
extract_contour_module() {
    log "开始提取contour特征模块..."
    
    cd "$PROJECT_DIR" || error_exit "无法进入项目目录"
    
    # 检查提取脚本是否存在
    if [ ! -f "extract_contour_module.py" ]; then
        error_exit "提取脚本不存在: extract_contour_module.py"
    fi
    
    # 执行提取，输出到日志文件
    python extract_contour_module.py > "$EXTRACT_LOG" 2>&1
    local extract_exit_code=$?
    
    if [ $extract_exit_code -ne 0 ]; then
        error_exit "模块提取失败，请查看日志: $EXTRACT_LOG"
    fi
    
    # 检查输出文件是否生成
    if [ ! -f "contour_feature_extractor_stage1_contour_extraction.pth" ]; then
        error_exit "提取完成但未找到输出文件"
    fi
    
    log "contour特征模块提取完成"
}

# 运行测试函数
run_tests() {
    log "开始运行测试..."
    
    cd "$PROJECT_DIR" || error_exit "无法进入项目目录"
    
    # 检查测试脚本是否存在
    if [ ! -f "test_contour_extractor.py" ]; then
        error_exit "测试脚本不存在: test_contour_extractor.py"
    fi
    
    # 执行测试，输出到日志文件
    python test_contour_extractor.py > "$TEST_LOG" 2>&1
    local test_exit_code=$?
    
    if [ $test_exit_code -ne 0 ]; then
        log "WARNING: 测试执行失败，请查看日志: $TEST_LOG"
    else
        log "测试执行完成"
    fi
}

# 清理函数
cleanup() {
    log "执行清理操作..."
    if [ -f "$PID_FILE" ]; then
        rm -f "$PID_FILE"
    fi
    log "清理完成"
}

# 主执行函数
main() {
    log "开始PairingNet Contour特征提取完整流程"
    log "日志文件: $MAIN_LOG"
    
    # 保存PID
    echo $$ > "$PID_FILE"
    
    # 设置信号处理
    trap cleanup EXIT
    trap 'log "收到中断信号，正在清理..."; cleanup; exit 130' INT TERM
    
    # 执行各个步骤
    check_dependencies
    check_dataset
    train_stage1
    extract_contour_module
    run_tests
    
    log "PairingNet Contour特征提取完整流程执行完成"
    log "所有日志文件保存在: $LOG_DIR"
    
    # 显示结果摘要
    echo "=========================================="
    echo "执行完成摘要:"
    echo "主日志: $MAIN_LOG"
    echo "训练日志: $TRAIN_LOG"
    echo "提取日志: $EXTRACT_LOG"
    echo "测试日志: $TEST_LOG"
    echo "输出文件: $PROJECT_DIR/contour_feature_extractor_stage1_contour_extraction.pth"
    echo "=========================================="
}

# 显示使用说明
usage() {
    cat << EOF
用法: $0 [选项]

选项:
    -h, --help          显示此帮助信息
    -b, --background    在后台运行脚本
    -s, --status        查看后台任务状态
    -k, --kill          终止后台任务
    -l, --logs          查看实时日志

示例:
    $0                  # 前台运行
    $0 -b               # 后台运行
    $0 -s               # 查看状态
    $0 -l               # 查看日志
    $0 -k               # 终止任务
EOF
}

# 查看状态函数
check_status() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo "后台任务正在运行 (PID: $pid)"
            echo "日志文件: $MAIN_LOG"
            return 0
        else
            echo "后台任务已停止"
            rm -f "$PID_FILE"
            return 1
        fi
    else
        echo "没有运行中的后台任务"
        return 1
    fi
}

# 终止任务函数
kill_task() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo "正在终止任务 (PID: $pid)..."
            kill -TERM "$pid"
            sleep 2
            if kill -0 "$pid" 2>/dev/null; then
                echo "强制终止任务..."
                kill -KILL "$pid"
            fi
            rm -f "$PID_FILE"
            echo "任务已终止"
        else
            echo "任务已停止"
            rm -f "$PID_FILE"
        fi
    else
        echo "没有运行中的任务"
    fi
}

# 查看日志函数
view_logs() {
    if [ -f "$MAIN_LOG" ]; then
        echo "实时查看日志 (Ctrl+C 退出):"
        tail -f "$MAIN_LOG"
    else
        echo "日志文件不存在"
    fi
}

# 参数解析
case "$1" in
    -h|--help)
        usage
        exit 0
        ;;
    -b|--background)
        echo "在后台启动PairingNet Contour特征提取..."
        nohup "$0" > /dev/null 2>&1 &
        echo "后台任务已启动"
        echo "使用 '$0 -s' 查看状态"
        echo "使用 '$0 -l' 查看日志"
        exit 0
        ;;
    -s|--status)
        check_status
        exit $?
        ;;
    -k|--kill)
        kill_task
        exit 0
        ;;
    -l|--logs)
        view_logs
        exit 0
        ;;
    "")
        # 无参数，正常执行
        main
        ;;
    *)
        echo "未知选项: $1"
        usage
        exit 1
        ;;
esac