# Run command in tmux session
function run_command_in_tmux {
    local session_name=$1
    local command=$2
    local wait_pattern=$3

    tmux send-keys -t "$session_name" "$command" Enter
    echo "runing command $command in session $session_name"
    if [ -n "$wait_pattern" ]; then
        echo "waiting for $wait_pattern in session $session_name"
        # Loop until the wait_pattern is found
        while true; do
            tmux capture-pane -t "$session_name" -p | grep "$wait_pattern" && break
            sleep 1
        done
    fi
}

function wait_for_finish {
    local session_name="$1"
    local command="$2"
    # 使用 pgrep 命令获取进程 ID
    pid_session=$(pgrep -f "$command")
    echo "Waiting for the processes '$pid_session' matching '$command' to finish in session $session_name."
    for pid in $pid_session; do
        while ps -p "$pid" > /dev/null
        do
            sleep 1
        done
    done
    echo "All processes matching '$command' in session $session_name have finished."
}

function monitor_session {
    local session_name=$1
    local wait_pattern=$2

    while true; do
        if tmux capture-pane -t "$session_name" -p | grep -q "$wait_pattern"; then
            echo "Found '$wait_pattern' in session $session_name."
            break
        fi

        sleep 1
    done
    echo "Exiting loop for session $session_name."
}

# 定义一个函数，用于管理tmux会话
manage_tmux_sessions() {
    # 接收一个包含会话编号的数组作为参数
    local session_numbers=("$@")

    # 遍历会话编号数组
    for session_number in "${session_numbers[@]}"; do
        # 检查会话是否存在
        if tmux has-session -t "$session_number" 2>/dev/null; then
            # 如果会话存在，发送命令到该会话
            tmux send-keys -t "$session_number" "conda activate pointnet" C-m
            tmux send-keys -t "$session_number" "cd /home/user_tp/workspace/code/attack/ModelNet40-C" C-m
            # 附加到该会话
            tmux attach-session -t "$session_number"
        else
            # 如果会话不存在，创建一个新的会话并命名
            tmux new-session -d -s "$session_number"
            # 发送命令到新创建的会话
            tmux send-keys -t "$session_number" "conda activate pointnet" C-m
            tmux send-keys -t "$session_number" "cd /home/user_tp/workspace/code/base_model/Pointnet_Pointnet2_pytorch-master" C-m
            # 由于是新创建的会话，所以立即附加进去
            tmux attach-session -t "$session_number"
        fi
    done
}

# 定义会话编号数组
session_numbers=(1 2 3 4 5 6 7 8)

tmux_session_1="1"
tmux_session_2="2"
tmux_session_3="3"
tmux_session_4="4"
tmux_session_5="5"
tmux_session_6="6"
tmux_session_7="7"
tmux_session_8="8"
tmux_session_9="9"
tmux_session_10="10"
tmux_session_11="11"

 
# for cor in 'scale' 'jitter' 'dropout_global' 'dropout_local' 'add_global' 'add_local' 'rotate'; do
# for cor in 'dropout_global' 'dropout_local' 'add_global' 'add_local'; do

# for sev in 0 1 2 3 4; do
#     # run_command_in_tmux "$tmux_session_3" "CUDA_VISIBLE_DEVICES=1 python main.py --entry pcc --model-path checkpoints/dgcnn_rpc_run_1_WOLFMix/model_final.t7 --exp-config configs/corruption/rpc.yaml  --severity ${sev} --corruption ${cor} --use_upsample no --sample_type no --add_prefix" "{'acc':"    
#     run_command_in_tmux "$tmux_session_3" "CUDA_VISIBLE_DEVICES=3 python main.py --entry pcc --model-path checkpoints/dgcnn_rpc_run_1_up_or_down_ratio_score_2_wrs/model_best_test.pth --exp-config configs/corruption/rpc.yaml --use_upsample median_hroup --sample_type ffps_0.95 --severity ${sev} --corruption ${cor}" "{'acc':"    
# done
# done

# for cor in 'occlusion' 'lidar' 'density_inc' 'density' 'cutout' 'uniform' 'gaussian' 'impulse' 'upsampling' 'background' 'rotation' 'shear' 'distortion' 'distortion_rbf' 'distortion_rbf_inv'; do
for cor in 'density' 'cutout'; do
for sev in 1 2 3 4 5; do
    # run_command_in_tmux "$tmux_session_3" "CUDA_VISIBLE_DEVICES=3 python main.py --entry mnc --model-path checkpoints/dgcnn_rpc_run_1_WOLFMix/model_final.t7 --exp-config configs/corruption/rpc.yaml --use_upsample no --sample_type no --severity ${sev} --corruption ${cor} --add_prefix" "{'acc':"    
    run_command_in_tmux "$tmux_session_3" "CUDA_VISIBLE_DEVICES=1 python main.py --entry mnc --model-path checkpoints/dgcnn_rpc_run_1_up_or_down_ratio_score_2_wrs/model_best_test.pth --exp-config configs/corruption/rpc.yaml --use_upsample median_hroup --sample_type ffps_0.95 --severity ${sev} --corruption ${cor}" "{'acc':"    
    
done
done
##model pcc
# for u in 'r_rwup' 'up_or_down'; do
# for d in 'wrs';do
# for up in 'no' 'r_rwup'; do 
# for down in 'wfps';do
# for cor in 'scale' 'jitter' 'dropout_global' 'dropout_local' 'add_global' 'add_local' 'rotate'; do
# for sev in 0 1 2 3 4; do
# run_command_in_tmux "$tmux_session_1" "CUDA_VISIBLE_DEVICES=0 python main.py --entry pcc --model-path runs/dgcnn_pct_run_1_${u}_${d}/model_best_test.pth --exp-config configs/corruption/pct.yaml --use_upsample ${up} --sample_type ${down} --severity ${sev} --corruption ${cor}" "{'acc':"
# run_command_in_tmux "$tmux_session_2" "CUDA_VISIBLE_DEVICES=1 python main.py --entry pcc --model-path runs/dgcnn_pointnet2_run_1_${u}_${d}/model_best_test.pth --exp-config configs/corruption/pointnet2.yaml --use_upsample ${up} --sample_type ${down} --severity ${sev} --corruption ${cor}" "{'acc':"
# done
# done
# done
# done
# done
# done


# for model in 'pct' 'pointnet2'; do #'gdanet' 'pointnet' 'pct' 'rscnn' 'pointnet2'  'simpleview' 'dgcnn'  'pointMLP' 'curvenet'; do

#     for aug in 'rsmix' 'cutmix_r' 'cutmix_k' 'mixup';do # 'pgd'; do

#         run_command_in_tmux "$tmux_session_1" "python main.py --entry mnc --model-path runs/${aug}_${model}_run_1/model_best_test.pth --exp-config configs/corruption/${model}.yaml" "{'acc':"
#         run_command_in_tmux "$tmux_session_2" "python main.py --entry test --model-path runs/${aug}_${model}_run_1/model_best_test.pth --exp-config configs/dgcnn_${model}_run_1.yaml" "{'acc':"
        
#     done
# done

        #         run_command_in_tmux "$tmux_session_1" "CUDA_VISIBLE_DEVICES=0 python main.py --entry mnc --model-path runs/dgcnn_pct_run_1_no_fps/model_best_test.pth --exp-config configs/corruption/pct.yaml --use_upsample ${up} --sample_type ${down}" "{'acc':"
#         run_command_in_tmux "$tmux_session_2" "CUDA_VISIBLE_DEVICES=0 python main.py --entry mnc --model-path runs/dgcnn_pct_run_1_no_wfps/model_best_test.pth --exp-config configs/corruption/pct.yaml --use_upsample ${up} --sample_type ${down}" "{'acc':"
#         run_command_in_tmux "$tmux_session_3" "CUDA_VISIBLE_DEVICES=3 python main.py --entry mnc --model-path runs/dgcnn_pct_run_1_no_wrs/model_best_test.pth --exp-config configs/corruption/pct.yaml --use_upsample ${up} --sample_type ${down}" "{'acc':"    
        
#         run_command_in_tmux "$tmux_session_4" "CUDA_VISIBLE_DEVICES=1 python main.py --entry mnc --model-path runs/dgcnn_pct_run_1_r_rwup_fps/model_best_test.pth --exp-config configs/corruption/pct.yaml --use_upsample ${up} --sample_type ${down}" "{'acc':"
#         run_command_in_tmux "$tmux_session_5" "CUDA_VISIBLE_DEVICES=1 python main.py --entry mnc --model-path runs/dgcnn_pct_run_1_r_rwup_wfps/model_best_test.pth --exp-config configs/corruption/pct.yaml --use_upsample ${up} --sample_type ${down}" "{'acc':"
#         run_command_in_tmux "$tmux_session_6" "CUDA_VISIBLE_DEVICES=1 python main.py --entry mnc --model-path runs/dgcnn_pct_run_1_r_rwup_wrs/model_best_test.pth --exp-config configs/corruption/pct.yaml --use_upsample ${up} --sample_type ${down}" "{'acc':"    
        
#         run_command_in_tmux "$tmux_session_7" "CUDA_VISIBLE_DEVICES=3 python main.py --entry mnc --model-path runs/dgcnn_pct_run_1_up_or_down_fps/model_best_test.pth --exp-config configs/corruption/pct.yaml --use_upsample ${up} --sample_type ${down}" "{'acc':"
#         run_command_in_tmux "$tmux_session_8" "CUDA_VISIBLE_DEVICES=3 python main.py --entry mnc --model-path runs/dgcnn_pct_run_1_up_or_down_wfps/model_best_test.pth --exp-config configs/corruption/pct.yaml --use_upsample ${up} --sample_type ${down}" "{'acc':"
#         run_command_in_tmux "$tmux_session_9" "CUDA_VISIBLE_DEVICES=3 python main.py --entry mnc --model-path runs/dgcnn_pct_run_1_up_or_down_wrs/model_best_test.pth --exp-config configs/corruption/pct.yaml --use_upsample ${up} --sample_type ${down}" "{'acc':"    
    
# for u in 'no' 'r_rwup' 'up_or_down'; do
# for d in 'fps' 'wfps' 'wrs';do
# for up in 'no' 'r_rwup'; do 
# for down in 'fps' 'wfps' 'wrs';do
# run_command_in_tmux "$tmux_session_11" "CUDA_VISIBLE_DEVICES=0 python main.py --entry test --model-path runs/dgcnn_pointnet2_run_1_${u}_${d}/model_best_test.pth --exp-config configs/dgcnn_pointnet2_run_1.yaml --use_upsample ${up} --sample_type ${down}" "{'acc':"
# done
# done
# done
# done
# done
# done

# run_command_in_tmux "$tmux_session_4" "CUDA_VISIBLE_DEVICES=1 python main.py --entry test --model-path runs/dgcnn_pct_run_1_r_rwup_fps/model_best_test.pth --exp-config configs/dgcnn_pct_run_1.yaml --use_upsample ${up} --sample_type ${down}" "{'acc':"
# run_command_in_tmux "$tmux_session_5" "CUDA_VISIBLE_DEVICES=1 python main.py --entry pcc --model-path runs/dgcnn_pct_run_1_r_rwup_wfps/model_best_test.pth --exp-config configs/corruption/pct.yaml --use_upsample ${up} --sample_type ${down}" "{'acc':"
# run_command_in_tmux "$tmux_session_6" "CUDA_VISIBLE_DEVICES=1 python main.py --entry mnc --model-path runs/dgcnn_pct_run_1_r_rwup_wrs/model_best_test.pth --exp-config configs/corruption/pct.yaml --use_upsample ${up} --sample_type ${down}" "{'acc':"    
        

# for up in 'no' 'r_rwup'; do 
#     for down in 'fps' 'wfps' 'wrs';do
#         run_command_in_tmux "$tmux_session_4" "CUDA_VISIBLE_DEVICES=1 python main.py --entry mnc --model-path runs/dgcnn_pointnet2_run_1_${up}_${down}/model_best_test.pth --exp-config configs/dgcnn_pointnet2_run_1.yaml --use_upsample ${up} --sample_type ${down}" "{'acc':"
#         run_command_in_tmux "$tmux_session_5" "CUDA_VISIBLE_DEVICES=1 python main.py --entry pcc --model-path runs/dgcnn_pointnet2_run_1_${up}_${down}/model_best_test.pth --exp-config configs/corruption/pointnet2.yaml --use_upsample ${up} --sample_type ${down}" "{'acc':"
#         run_command_in_tmux "$tmux_session_6" "CUDA_VISIBLE_DEVICES=1 python main.py --entry test --model-path runs/dgcnn_pointnet2_run_1_${up}_${down}/model_best_test.pth --exp-config configs/corruption/pointnet2.yaml --use_upsample ${up} --sample_type ${down}" "{'acc':"    
#     done
# done

# CUDA_VISIBLE_DEVICES=0 python main.py --entry test --model-path runs/${aug}_${model}_run_1/model_best_test.pth --exp-config configs/corruption/${model}.yaml

# done

# command_train_0="CUDA_VISIBLE_DEVICES=3 python main.py --exp-config configs/dgcnn_pointnet2_run_1.yaml"
# command_train_1="CUDA_VISIBLE_DEVICES=3 python main.py --exp-config configs/cutmix/pointnet2_r.yaml"
# command_train_2="CUDA_VISIBLE_DEVICES=3 python main.py --exp-config configs/cutmix/pointnet2_k.yaml"
# command_train_3="CUDA_VISIBLE_DEVICES=3 python main.py --exp-config configs/mixup/pointnet2.yaml"
# command_train_4="CUDA_VISIBLE_DEVICES=3 python main.py --exp-config configs/rsmix/pointnet2.yaml"

# command_test_5="CUDA_VISIBLE_DEVICES=0 python main.py --entry test --model-path runs/dgcnn_pointnet2_run_1_r_rwup_wrs/model_best_test.pth --exp-config configs/dgcnn_pointnet2_run_1.yaml --use_upsample r_rwup --sample_type wrs"
# command_test_6="CUDA_VISIBLE_DEVICES=2 python main.py --entry test --model-path runs/dgcnn_pointnet2_run_1_up_or_down_fps/model_best_test.pth --exp-config configs/dgcnn_pointnet2_run_1.yaml --use_upsample up_or_down"
# command_test_7="CUDA_VISIBLE_DEVICES=3 python main.py --entry test --model-path runs/dgcnn_pointnet2_run_1_up_or_down_wfps/model_best_test.pth --exp-config configs/dgcnn_pointnet2_run_1.yaml --use_upsample up_or_down --sample_type wfps"

# command_test_0="CUDA_VISIBLE_DEVICES=3 python main.py --entry test --model-path runs/dgcnn_pointnet2_run_1/model_best_test.pth --exp-config configs/dgcnn_pointnet2_run_1.yaml"
# command_test_1="CUDA_VISIBLE_DEVICES=3 python main.py --entry test --model-path runs/cutmix_r_pointnet2_run_1/model_best_test.pth --exp-config configs/cutmix/pointnet2_r.yaml"
# command_test_2="CUDA_VISIBLE_DEVICES=3 python main.py --entry test --model-path runs/cutmix_k_pointnet2_run_1/model_best_test.pth --exp-config configs/cutmix/pointnet2_k.yaml"
# command_test_3="CUDA_VISIBLE_DEVICES=3 python main.py --entry test --model-path runs/mixup_pointnet2_run_1/model_best_test.pth --exp-config configs/mixup/pointnet2.yaml"
# command_test_4="CUDA_VISIBLE_DEVICES=3 python main.py --entry test --model-path runs/rsmix_pointnet2_run_1/model_best_test.pth --exp-config configs/rsmix/pointnet2.yaml"

# 'dgcnn_pct_run_1_no_fps','dgcnn_pointnet2_run_1_no_fps',
# 'dgcnn_pct_run_1_no_wfps','dgcnn_pct_run_1_no_wrs','dgcnn_pct_run_1_r_rwup_fps','dgcnn_pct_run_1_r_rwup_wfps',
# 'dgcnn_pct_run_1_r_rwup_wrs','dgcnn_pct_run_1_up_or_down_fps','dgcnn_pct_run_1_up_or_down_wfps','dgcnn_pct_run_1_up_or_down_wrs',
# 'dgcnn_pointnet2_run_1_no_wfps','dgcnn_pointnet2_run_1_no_wrs','dgcnn_pointnet2_run_1_r_rwup_fps',
# 'dgcnn_pointnet2_run_1_r_rwup_wfps','dgcnn_pointnet2_run_1_up_or_down_wrs'
# 'dgcnn_pointnet2_run_1_r_rwup_wrs','dgcnn_pointnet2_run_1_up_or_down_fps','dgcnn_pointnet2_run_1_up_or_down_wfps'

# command_testc_5="CUDA_VISIBLE_DEVICES=2 python main.py --entry mnc --model-path runs/dgcnn_pointnet2_run_1_r_rwup_wrs/model_best_test.pth --exp-config configs/corruption/pointnet2.yaml --use_upsample r_rwup --sample_type wrs"
# command_testc_6="CUDA_VISIBLE_DEVICES=2 python main.py --entry mnc --model-path runs/dgcnn_pointnet2_run_1_up_or_down_fps/model_best_test.pth --exp-config configs/corruption/pointnet2.yaml --use_upsample up_or_down"
# command_testc_7="CUDA_VISIBLE_DEVICES=2 python main.py --entry mnc --model-path runs/dgcnn_pointnet2_run_1_up_or_down_wfps/model_best_test.pth --exp-config configs/corruption/pointnet2.yaml --use_upsample up_or_down --sample_type wfps"

#pct pcc
# command_testc_8="CUDA_VISIBLE_DEVICES=3 python main.py --entry pcc --model-path runs/dgcnn_pct_run_1_no_fps/model_best_test.pth --exp-config configs/corruption/pct.yaml"
# command_testc_0="CUDA_VISIBLE_DEVICES=0 python main.py --entry pcc --model-path runs/dgcnn_pct_run_1_no_wfps/model_best_test.pth --exp-config configs/corruption/pct.yaml --sample_type wfps"
# command_testc_1="CUDA_VISIBLE_DEVICES=1 python main.py --entry pcc --model-path runs/dgcnn_pct_run_1_no_wrs/model_best_test.pth --exp-config configs/corruption/pct.yaml --sample_type wrs"
# command_testc_2="CUDA_VISIBLE_DEVICES=2 python main.py --entry pcc --model-path runs/dgcnn_pct_run_1_r_rwup_fps/model_best_test.pth --exp-config configs/corruption/pct.yaml --use_upsample r_rwup"
# command_testc_3="CUDA_VISIBLE_DEVICES=3 python main.py --entry pcc --model-path runs/dgcnn_pct_run_1_r_rwup_wfps/model_best_test.pth --exp-config configs/corruption/pct.yaml --use_upsample r_rwup --sample_type wfps"
# command_testc_4="CUDA_VISIBLE_DEVICES=0 python main.py --entry pcc --model-path runs/dgcnn_pct_run_1_r_rwup_wrs/model_best_test.pth --exp-config configs/corruption/pct.yaml --use_upsample r_rwup --sample_type wrs"
# command_testc_5="CUDA_VISIBLE_DEVICES=1 python main.py --entry pcc --model-path runs/dgcnn_pct_run_1_up_or_down_fps/model_best_test.pth --exp-config configs/corruption/pct.yaml --use_upsample up_or_down"
# command_testc_6="CUDA_VISIBLE_DEVICES=2 python main.py --entry pcc --model-path runs/dgcnn_pct_run_1_up_or_down_wfps/model_best_test.pth --exp-config configs/corruption/pct.yaml --use_upsample up_or_down --sample_type wfps"
# command_testc_7="CUDA_VISIBLE_DEVICES=3 python main.py --entry pcc --model-path runs/dgcnn_pct_run_1_up_or_down_wrs/model_best_test.pth --exp-config configs/corruption/pct.yaml --use_upsample up_or_down --sample_type wrs"

#data augmentation
# command_testc_0="CUDA_VISIBLE_DEVICES=0 python main.py --entry pcc --model-path runs/cutmix_k_pct_run_1/model_best_test.pth --exp-config configs/cutmix/pct_k.yaml"
# command_testc_1="CUDA_VISIBLE_DEVICES=1 python main.py --entry pcc --model-path runs/cutmix_r_pct_run_1/model_best_test.pth --exp-config configs/cutmix/pct_r.yaml"
# command_testc_2="CUDA_VISIBLE_DEVICES=2 python main.py --entry pcc --model-path runs/mixup_pct_run_1/model_best_test.pth --exp-config configs/mixup/pct.yaml"
# command_testc_3="CUDA_VISIBLE_DEVICES=3 python main.py --entry pcc --model-path runs/pgd_pct_run_1/model_best_test.pth --exp-config configs/pgd/pct.yaml"
# command_testc_4="CUDA_VISIBLE_DEVICES=0 python main.py --entry pcc --model-path runs/rsmix_k_pct_run_1/model_best_test.pth --exp-config configs/rsmix/pct.yaml"
# command_testc_5="CUDA_VISIBLE_DEVICES=1 python main.py --entry pcc --model-path runs/cutmix_k_pointent2_run_1/model_best_test.pth --exp-config configs/cutmix/pointnet2_k.yaml"
# command_testc_6="CUDA_VISIBLE_DEVICES=2 python main.py --entry pcc --model-path runs/cutmix_r_pointent2_run_1/model_best_test.pth --exp-config configs/cutmix/pointnet2_k.yaml"
# command_testc_7="CUDA_VISIBLE_DEVICES=3 python main.py --entry pcc --model-path runs/mixup_pointent2_run_1/model_best_test.pth --exp-config configs/mixup/pointnet2.yaml"
# command_testc_8="CUDA_VISIBLE_DEVICES=3 python main.py --entry pcc --model-path runs/rsmix_k_pointent2_run_1/model_best_test.pth --exp-config configs/rsmix/pointnet2.yaml"

#pointnet2 pcc
# command_testc_0="CUDA_VISIBLE_DEVICES=0 python main.py --entry pcc --model-path runs/dgcnn_pointnet2_run_1_no_wfps/model_best_test.pth --exp-config configs/corruption/pointnet2.yaml --sample_type wfps"
# command_testc_1="CUDA_VISIBLE_DEVICES=1 python main.py --entry pcc --model-path runs/dgcnn_pointnet2_run_1_no_wfps/model_best_test.pth --exp-config configs/corruption/pointnet2.yaml --sample_type wfps"
# command_testc_2="CUDA_VISIBLE_DEVICES=2 python main.py --entry pcc --model-path runs/dgcnn_pointnet2_run_1_no_wrs/model_best_test.pth --exp-config configs/corruption/pointnet2.yaml --sample_type wrs"
# command_testc_3="CUDA_VISIBLE_DEVICES=3 python main.py --entry pcc --model-path runs/dgcnn_pointnet2_run_1_r_rwup_fps/model_best_test.pth --exp-config configs/corruption/pointnet2.yaml --use_upsample r_rwup"
# command_testc_4="CUDA_VISIBLE_DEVICES=0 python main.py --entry pcc --model-path runs/dgcnn_pointnet2_run_1_r_rwup_wfps/model_best_test.pth --exp-config configs/corruption/pointnet2.yaml --use_upsample r_rwup --sample_type wfps"
# command_testc_5="CUDA_VISIBLE_DEVICES=1 python main.py --entry mnc --model-path runs/dgcnn_pointnet2_run_1_r_rwup_wrs/model_best_test.pth --exp-config configs/corruption/pointnet2.yaml --use_upsample r_rwup --sample_type wrs"
# command_testc_6="CUDA_VISIBLE_DEVICES=2 python main.py --entry mnc --model-path runs/dgcnn_pointnet2_run_1_up_or_down_fps/model_best_test.pth --exp-config configs/corruption/pointnet2.yaml --use_upsample up_or_down"
# command_testc_7="CUDA_VISIBLE_DEVICES=3 python main.py --entry mnc --model-path runs/dgcnn_pointnet2_run_1_up_or_down_wfps/model_best_test.pth --exp-config configs/corruption/pointnet2.yaml --use_upsample up_or_down --sample_type wfps"
# command_testc_8="CUDA_VISIBLE_DEVICES=0 python main.py --entry pcc --model-path runs/dgcnn_pointnet2_run_1_up_or_down_wrs/model_best_test.pth --exp-config configs/corruption/pointnet2.yaml --use_upsample up_or_down --sample_type wrs"

# run_command_in_tmux "$tmux_session_1" "$command_testc_0" "{'acc':"
# run_command_in_tmux "$tmux_session_2" "$command_testc_1" "{'acc':"
# run_command_in_tmux "$tmux_session_3" "$command_testc_2" "{'acc':"
# run_command_in_tmux "$tmux_session_4" "$command_testc_3" "{'acc':"
# run_command_in_tmux "$tmux_session_5" "$command_testc_4" "{'acc':"
# run_command_in_tmux "$tmux_session_1" "$command_test_5" "{'acc':"
# run_command_in_tmux "$tmux_session_9" "$command_test_6" "{'acc':"
# run_command_in_tmux "$tmux_session_10" "$command_test_7" "{'acc':"
# run_command_in_tmux "$tmux_session_9" "$command_testc_8" "{'acc':"

# command_testc_0="python main.py --entry mnc --model-path cor_exp/modify/rnum/pct_fps/model.t7 --exp-config configs/corruption/pct.yaml --use_upsample up_or_down --add_prefix --gpu 0"
# command_testc_1="python main.py --entry mnc --model-path cor_exp/modify/rnum/pct_wfps_t20/model.t7 --exp-config configs/corruption/pct.yaml --use_upsample up_or_down --sample_type wfps --add_prefix --gpu 0"
# command_testc_2="python main.py --entry mnc --model-path cor_exp/modify/up/pct_oup_fps/model.t7 --exp-config configs/corruption/pct.yaml --use_upsample oup --add_prefix --gpu 0"
# command_testc_3="python main.py --entry mnc --model-path cor_exp/modify/up/pct_roup_fps/model.t7 --exp-config configs/corruption/pct.yaml --use_upsample r_oup --add_prefix --gpu 0"
# command_testc_4="python main.py --entry mnc --model-path cor_exp/modify/up/pct_rrwup_fps/model.t7 --exp-config configs/corruption/pct.yaml --use_upsample r_rwup --add_prefix --gpu 1"
# command_testc_5="python main.py --entry mnc --model-path cor_exp/modify/up/pct_rwup_fps/model.t7 --exp-config configs/corruption/pct.yaml --use_upsample rwup --add_prefix --gpu 1"
# command_testc_6="python main.py --entry mnc --model-path cor_exp/modify/up_wfps/pct_oup_wfps_t20/model.t7 --exp-config configs/corruption/pct.yaml --use_upsample oup --sample_type wfps --add_prefix --gpu 1"
# command_testc_7="python main.py --entry mnc --model-path cor_exp/modify/up_wfps/pct_roup_wfps_t20/model.t7 --exp-config configs/corruption/pct.yaml --use_upsample r_oup --sample_type wfps --add_prefix --gpu 1"

# command_test_0="CUDA_VISIBLE_DEVICES=1 python main.py --entry test --model-path cor_exp/modify/rnum/pct_fps/model.t7 --exp-config configs/dgcnn_pct_run_1.yaml --use_upsample up_or_down --add_prefix --gpu 3"
# command_test_1="CUDA_VISIBLE_DEVICES=1 python main.py --entry test --model-path cor_exp/modify/rnum/pct_wfps_t20/model.t7 --exp-config configs/dgcnn_pct_run_1.yaml --use_upsample up_or_down --sample_type wfps --add_prefix --gpu 3"
# command_test_2="CUDA_VISIBLE_DEVICES=1 python main.py --entry test --model-path cor_exp/modify/up/pct_oup_fps/model.t7 --exp-config configs/dgcnn_pct_run_1.yaml --use_upsample oup --add_prefix --gpu 3"
# command_test_3="CUDA_VISIBLE_DEVICES=1 python main.py --entry test --model-path cor_exp/modify/up/pct_roup_fps/model.t7 --exp-config configs/dgcnn_pct_run_1.yaml --use_upsample r_oup --add_prefix --gpu 3"
# command_test_4="CUDA_VISIBLE_DEVICES=2 python main.py --entry test --model-path cor_exp/modify/up/pct_rrwup_fps/model.t7 --exp-config configs/dgcnn_pct_run_1.yaml --use_upsample r_rwup --add_prefix --gpu 2"
# command_test_5="CUDA_VISIBLE_DEVICES=2 python main.py --entry test --model-path cor_exp/modify/up/pct_rwup_fps/model.t7 --exp-config configs/dgcnn_pct_run_1.yaml --use_upsample rwup --add_prefix --gpu 2"
# command_test_6="CUDA_VISIBLE_DEVICES=2 python main.py --entry test --model-path cor_exp/modify/up_wfps/pct_oup_wfps_t20/model.t7 --exp-config configs/dgcnn_pct_run_1.yaml --use_upsample oup --sample_type wfps --add_prefix --gpu 2"
# command_test_7="CUDA_VISIBLE_DEVICES=2 python main.py --entry test --model-path cor_exp/modify/up_wfps/pct_roup_wfps_t20/model.t7 --exp-config configs/dgcnn_pct_run_1.yaml --use_upsample r_oup --sample_type wfps --add_prefix --gpu 2"


# 调用函数并传入会话编号数组
# manage_tmux_sessions "${session_numbers[@]}"


#Run commands_testc
# run_command_in_tmux "$tmux_session_1" "$command_train_0" "Testing.."
# run_command_in_tmux "$tmux_session_2" "$command_train_1" "Testing.."
# run_command_in_tmux "$tmux_session_3" "$command_train_2" "Testing.."
# run_command_in_tmux "$tmux_session_4" "$command_train_3" "Testing.."
# run_command_in_tmux "$tmux_session_5" "$command_train_4" "Testing.."


# Run commands_test
# wait_for_finish "tmux_session_1" "command_train_0"
# run_command_in_tmux "$tmux_session_1" "$command_test_0" "{'acc':"


# wait_for_finish "tmux_session_2" "command_train_1"
# run_command_in_tmux "$tmux_session_2" "$command_test_1" "{'acc':"

# wait_for_finish "tmux_session_3" "command_train_2"
# run_command_in_tmux "$tmux_session_3" "$command_test_2" "{'acc':"

# wait_for_finish "tmux_session_4" "command_train_3"
# run_command_in_tmux "$tmux_session_4" "$command_test_3" "{'acc':"


# wait_for_finish "tmux_session_5" "command_train_4"
# run_command_in_tmux "$tmux_session_5" "$command_test_4" "{'acc':"





# wait_for_finish "tmux_session_8" "command_train_7"
# for i in {1..5};do
#     run_command_in_tmux "$tmux_session_1" "$command_test_7" "{'acc':"
# done



echo "All commands executed successfully."
