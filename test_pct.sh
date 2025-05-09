###ModelNet-C by jiachen-sun,15
# MAPC = ['occlusion','lidar','density_inc','density','cutout','uniform','gaussian','impulse','upsampling',
        # 'background','rotation','shear','distortion','distortion_rbf','distortion_rbf_inv']
###modelnet-c by jiawei-ren(pointcloud-c),7
# MAPc = ['scale','jitter','dropout_global','dropout_local','add_global','add_local','rotate']
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

# for model in 'pct' 'pointnet2'; do
# for aug in 'rsmix' 'cutmix_r' 'cutmix_k' 'mixup' 'pgd'; do
# for cor in 'scale' 'jitter' 'dropout_global' 'dropout_local' 'add_global' 'add_local' 'rotate'; do
# for sev in 0 1 2 3 4; do
# run_command_in_tmux "$tmux_session_1" "CUDA_VISIBLE_DEVICES=0 python main.py --entry pcc --model-path runs/rsmix_pct_run_1/model_best_test.pth --exp-config configs/corruption/pct.yaml --severity ${sev} --corruption ${cor}" "{'acc':"
# run_command_in_tmux "$tmux_session_2" "CUDA_VISIBLE_DEVICES=0 python main.py --entry pcc --model-path runs/cutmix_k_pct_run_1/model_best_test.pth --exp-config configs/corruption/pct.yaml --severity ${sev} --corruption ${cor}" "{'acc':"
# run_command_in_tmux "$tmux_session_3" "CUDA_VISIBLE_DEVICES=1 python main.py --entry pcc --model-path runs/cutmix_r_pct_run_1/model_best_test.pth --exp-config configs/corruption/pct.yaml --severity ${sev} --corruption ${cor}" "{'acc':"
# run_command_in_tmux "$tmux_session_4" "CUDA_VISIBLE_DEVICES=1 python main.py --entry pcc --model-path runs/mixup_pct_run_1/model_best_test.pth --exp-config configs/corruption/pct.yaml --severity ${sev} --corruption ${cor}" "{'acc':"
# run_command_in_tmux "$tmux_session_5" "CUDA_VISIBLE_DEVICES=2 python main.py --entry pcc --model-path runs/pgd_pct_run_1/model_best_test.pth --exp-config configs/corruption/pct.yaml --severity ${sev} --corruption ${cor}" "{'acc':"

# run_command_in_tmux "$tmux_session_6" "CUDA_VISIBLE_DEVICES=2 python main.py --entry pcc --model-path runs/rsmix_pointnet2_run_1/model_best_test.pth --exp-config configs/corruption/pointnet2.yaml --severity ${sev} --corruption ${cor}" "{'acc':"
# run_command_in_tmux "$tmux_session_7" "CUDA_VISIBLE_DEVICES=3 python main.py --entry pcc --model-path runs/cutmix_k_pointnet2_run_1/model_best_test.pth --exp-config configs/corruption/pointnet2.yaml --severity ${sev} --corruption ${cor}" "{'acc':"
# run_command_in_tmux "$tmux_session_8" "CUDA_VISIBLE_DEVICES=3 python main.py --entry pcc --model-path runs/cutmix_r_pointnet2_run_1/model_best_test.pth --exp-config configs/corruption/pointnet2.yaml --severity ${sev} --corruption ${cor}" "{'acc':"
# run_command_in_tmux "$tmux_session_9" "CUDA_VISIBLE_DEVICES=3 python main.py --entry pcc --model-path runs/mixup_pointnet2_run_1/model_best_test.pth --exp-config configs/corruption/pointnet2.yaml --severity ${sev} --corruption ${cor}" "{'acc':"

# done
# done

# for up in 'median_hroup'; do
# for up in 'no'; do
# # for down in 'ffps_0.95'; do
# for down in 'fps'; do
#     run_command_in_tmux "$tmux_session_1" "CUDA_VISIBLE_DEVICES=3 python main.py --entry test --model-path runs/pgd_curvenet_run_1_no_fps/model_best_test.pth --exp-config configs/dgcnn_curvenet_run_1.yaml --use_upsample ${up} --sample_type ${down}" "{'acc':"    
#     run_command_in_tmux "$tmux_session_2" "CUDA_VISIBLE_DEVICES=3 python main.py --entry oo3 --model-path runs/pgd_curvenet_run_1_no_fps/model_best_test.pth --exp-config configs/dgcnn_curvenet_run_1.yaml --use_upsample ${up} --sample_type ${down}" "{'acc':"    
#     # run_command_in_tmux "$tmux_session_3" "CUDA_VISIBLE_DEVICES=3 python main.py --entry test --model-path runs/dgcnn_pct_run_1_no_wrs/model_best_test.pth --exp-config configs/dgcnn_pct_run_1.yaml --use_upsample ${up} --sample_type ${down}" "{'acc':"    
#     # run_command_in_tmux "$tmux_session_4" "CUDA_VISIBLE_DEVICES=3 python main.py --entry test --model-path runs/rsmix_curvenet_run_1_no_fps/model_best_test.pth --exp-config configs/dgcnn_curvenet_run_1.yaml --use_upsample ${up} --sample_type ${down}" "{'acc':"    
#     # run_command_in_tmux "$tmux_session_5" "CUDA_VISIBLE_DEVICES=3 python main.py --entry test --model-path runs/mixup_curvenet_run_1_no_fps/model_best_test.pth --exp-config configs/dgcnn_curvenet_run_1.yaml --use_upsample ${up} --sample_type ${down}" "{'acc':"    
#     # run_command_in_tmux "$tmux_session_2" "CUDA_VISIBLE_DEVICES=3 python main.py --entry test --model-path runs/rsmix_gdanet_run_1_no_no/model_best_test.pth --exp-config configs/dgcnn_gdanet_run_1.yaml --use_upsample ${up} --sample_type ${down}" "{'acc':"    
    
# for cor in 'scale' 'jitter' 'dropout_global' 'dropout_local' 'add_global' 'add_local' 'rotate'; do 
# # for cor in 'rotate'; do 
# for sev in 0 1 2 3 4; do
#     run_command_in_tmux "$tmux_session_1" "CUDA_VISIBLE_DEVICES=3 python main.py --entry pcc --model-path runs/pgd_curvenet_run_1_no_fps/model_best_test.pth --exp-config configs/corruption/curvenet.yaml --use_upsample ${up} --sample_type ${down} --severity ${sev} --corruption ${cor}" "{'acc':"    
#     # run_command_in_tmux "$tmux_session_2" "CUDA_VISIBLE_DEVICES=3 python main.py --entry pcc --model-path runs/pgd_curvenet_run_1_no_fps/model_best_test.pth --exp-config configs/corruption/curvenet.yaml --use_upsample ${up} --sample_type ${down} --severity ${sev} --corruption ${cor}" "{'acc':"    
#     # run_command_in_tmux "$tmux_session_3" "CUDA_VISIBLE_DEVICES=3 python main.py --entry pcc --model-path runs/dgcnn_pct_run_1_no_wrs/model_best_test.pth --exp-config configs/corruption/pct.yaml --use_upsample ${up} --sample_type ${down} --severity ${sev} --corruption ${cor}" "{'acc':"    
#     # run_command_in_tmux "$tmux_session_4" "CUDA_VISIBLE_DEVICES=3 python main.py --entry pcc --model-path runs/rsmix_curvenet_run_1_no_fps/model_best_test.pth --exp-config configs/corruption/curvenet.yaml --use_upsample ${up} --sample_type ${down} --severity ${sev} --corruption ${cor}" "{'acc':"    
#     # run_command_in_tmux "$tmux_session_5" "CUDA_VISIBLE_DEVICES=3 python main.py --entry pcc --model-path runs/mixup_curvenet_run_1_no_fps/model_best_test.pth --exp-config configs/corruption/curvenet.yaml --use_upsample ${up} --sample_type ${down} --severity ${sev} --corruption ${cor}" "{'acc':"    
#     # run_command_in_tmux "$tmux_session_8" "CUDA_VISIBLE_DEVICES=3 python main.py --entry pcc --model-path runs/dgcnn_pct_run_1_r_rwup_wrs/model_best_test.pth --exp-config configs/corruption/pct.yaml --severity ${sev} --corruption ${cor} --use_upsample ${up} --sample_type ${down}" "{'acc':"    
# done
# done

# for cor in 'scale' 'jitter' 'dropout_global' 'dropout_local' 'add_global' 'add_local' 'rotate'; do 
# # # for cor in 'rotate'; do 
# for sev in 0 1 2 3 4; do
#     run_command_in_tmux "$tmux_session_1" "CUDA_VISIBLE_DEVICES=3 python main.py --entry oo3dc --model-path runs/pgd_curvenet_run_1_no_fps/model_best_test.pth --exp-config configs/corruption/curvenet.yaml --use_upsample ${up} --sample_type ${down} --severity ${sev} --corruption ${cor}" "{'acc':"    
#     # run_command_in_tmux "$tmux_session_2" "CUDA_VISIBLE_DEVICES=3 python main.py --entry oo3dc --model-path runs/pgd_curvenet_run_1_no_fps/model_best_test.pth --exp-config configs/corruption/curvenet.yaml --use_upsample ${up} --sample_type ${down} --severity ${sev} --corruption ${cor}" "{'acc':"    
# #     run_command_in_tmux "$tmux_session_3" "CUDA_VISIBLE_DEVICES=3 python main.py --entry oo3dc --model-path runs/cutmix_k_curvenet_run_1_no_fps/model_best_test.pth --exp-config configs/corruption/curvenet.yaml --use_upsample ${up} --sample_type ${down} --severity ${sev} --corruption ${cor}" "{'acc':"    
# #     run_command_in_tmux "$tmux_session_4" "CUDA_VISIBLE_DEVICES=3 python main.py --entry oo3dc --model-path runs/rsmix_curvenet_run_1_no_fps/model_best_test.pth --exp-config configs/corruption/curvenet.yaml --use_upsample ${up} --sample_type ${down} --severity ${sev} --corruption ${cor}" "{'acc':"    
# #     run_command_in_tmux "$tmux_session_5" "CUDA_VISIBLE_DEVICES=3 python main.py --entry oo3dc --model-path runs/mixup_curvenet_run_1_no_fps/model_best_test.pth --exp-config configs/corruption/curvenet.yaml --use_upsample ${up} --sample_type ${down} --severity ${sev} --corruption ${cor}" "{'acc':"    
# #     # run_command_in_tmux "$tmux_session_8" "CUDA_VISIBLE_DEVICES=3 python main.py --entry pcc --model-path runs/dgcnn_pct_run_1_r_rwup_wrs/model_best_test.pth --exp-config configs/corruption/pct.yaml --severity ${sev} --corruption ${cor} --use_upsample ${up} --sample_type ${down}" "{'acc':"    
# done
# done

# # 'occlusion' 'lidar' 'density_inc' 'density' 
# for cor in 'occlusion' 'lidar' 'density_inc' 'density' 'cutout' 'uniform' 'gaussian' 'impulse' 'upsampling' 'background' 'rotation' 'shear' 'distortion' 'distortion_rbf' 'distortion_rbf_inv'; do
# for sev in 1 2 3 4 5; do

#     run_command_in_tmux "$tmux_session_1" "CUDA_VISIBLE_DEVICES=3 python main.py --entry mnc --model-path runs/pgd_curvenet_run_1_no_fps/model_best_test.pth --exp-config configs/corruption/curvenet.yaml --use_upsample ${up} --sample_type ${down} --severity ${sev} --corruption ${cor}" "{'acc':"    
#     # run_command_in_tmux "$tmux_session_2" "CUDA_VISIBLE_DEVICES=3 python main.py --entry mnc --model-path runs/dgcnn_pct_run_1_up_or_down_ratio_score_2_ffps_0.95/model_best_test.pth --exp-config configs/corruption/pct.yaml --use_upsample ${up} --sample_type ${down} --severity ${sev} --corruption ${cor}" "{'acc':"    
#     # run_command_in_tmux "$tmux_session_3" "CUDA_VISIBLE_DEVICES=3 python main.py --entry mnc --model-path runs/dgcnn_pct_run_1_no_wrs/model_best_test.pth --exp-config configs/corruption/pct.yaml --use_upsample ${up} --sample_type ${down} --severity ${sev} --corruption ${cor}" "{'acc':"    
#     #     run_command_in_tmux "$tmux_session_5" "CUDA_VISIBLE_DEVICES=0 python main.py --entry mnc --model-path runs/dgcnn_pct_run_1_up_or_down_0.1_wrs/model_best_test.pth --exp-config configs/corruption/pct.yaml --severity ${sev} --corruption ${cor} --use_upsample ${up} --sample_type ${down}" "{'acc':"    
#     #     run_command_in_tmux "$tmux_session_6" "CUDA_VISIBLE_DEVICES=1 python main.py --entry mnc --model-path runs/dgcnn_pct_run_1_up_or_kdown_0.2_wrs/model_best_test.pth --exp-config configs/corruption/pct.yaml --severity ${sev} --corruption ${cor} --use_upsample ${up} --sample_type ${down}" "{'acc':"    
#     #     run_command_in_tmux "$tmux_session_7" "CUDA_VISIBLE_DEVICES=2 python main.py --entry mnc --model-path runs/dgcnn_pct_run_1_up_or_kdown_0.3_wrs/model_best_test.pth --exp-config configs/corruption/pct.yaml --severity ${sev} --corruption ${cor} --use_upsample ${up} --sample_type ${down}" "{'acc':"    
#     #     run_command_in_tmux "$tmux_session_8" "CUDA_VISIBLE_DEVICES=3 python main.py --entry mnc --model-path runs/dgcnn_pct_run_1_r_rwup_wrs/model_best_test.pth --exp-config configs/corruption/pct.yaml --severity ${sev} --corruption ${cor} --use_upsample ${up} --sample_type ${down}" "{'acc':"    
# done
# done
# done
# done

# for up in 'no'; do
# for down in 'ffps_0.95'; do
# # for down in 'fps'; do
#     # run_command_in_tmux "$tmux_session_3" "CUDA_VISIBLE_DEVICES=0 python main.py --entry test --model-path runs/dgcnn_curvenet_run_1_up_or_down_ratio_score_2_wrs/model_final.pth --exp-config configs/dgcnn_curvenet_run_1.yaml --use_upsample ${up} --sample_type ${down}" "{'acc':"    
#     # run_command_in_tmux "$tmux_session_2" "CUDA_VISIBLE_DEVICES=3 python main.py --entry test --model-path runs/dgcnn_curvenet_run_1_up_or_down_ratio_score_2_wrs/model_250.pth --exp-config configs/dgcnn_curvenet_run_1.yaml --use_upsample ${up} --sample_type ${down}" "{'acc':"    
#     # run_command_in_tmux "$tmux_session_3" "CUDA_VISIBLE_DEVICES=3 python main.py --entry oo3 --model-path runs/dgcnn_curvenet_run_1_up_or_down_ratio_score_2_wrs/model_275.pth --exp-config configs/dgcnn_curvenet_run_1.yaml --use_upsample ${up} --sample_type ${down}" "{'acc':"    
#     # run_command_in_tmux "$tmux_session_4" "CUDA_VISIBLE_DEVICES=0 python main.py --entry test --model-path runs/dgcnn_curvenet_run_1_up_or_down_ratio_score_2_wrs/model_275.pth --exp-config configs/dgcnn_curvenet_run_1.yaml --use_upsample ${up} --sample_type ${down}" "{'acc':"    
#     # run_command_in_tmux "$tmux_session_5" "CUDA_VISIBLE_DEVICES=2 python main.py --entry oo3 --model-path runs/dgcnn_curvenet_run_1_up_or_down_ratio_score_2_wrs/model_final.pth --exp-config configs/dgcnn_curvenet_run_1.yaml --use_upsample ${up} --sample_type ${down}" "{'acc':"    
#     # run_command_in_tmux "$tmux_session_6" "CUDA_VISIBLE_DEVICES=0 python main.py --entry test --model-path runs/dgcnn_curvenet_run_1_up_or_down_ratio_score_2_wrs/model_final.pth --exp-config configs/dgcnn_curvenet_run_1.yaml --use_upsample ${up} --sample_type ${down}" "{'acc':"    

# for cor in 'scale' 'jitter' 'dropout_global' 'dropout_local' 'add_global' 'add_local' 'rotate'; do 
# # # for cor in 'rotate'; do 
# for sev in 0 1 2 3 4; do
#     # run_command_in_tmux "$tmux_session_1" "CUDA_VISIBLE_DEVICES=3 python main.py --entry pcc --model-path runs/dgcnn_curvenet_run_1_up_or_down_ratio_score_2_wrs/model_250.pth --exp-config configs/corruption/curvenet.yaml --use_upsample ${up} --sample_type ${down} --severity ${sev} --corruption ${cor}" "{'acc':"    
#     # run_command_in_tmux "$tmux_session_2" "CUDA_VISIBLE_DEVICES=3 python main.py --entry pcc --model-path runs/dgcnn_curvenet_run_1_up_or_down_ratio_score_2_wrs/model_275.pth --exp-config configs/corruption/curvenet.yaml --use_upsample ${up} --sample_type ${down} --severity ${sev} --corruption ${cor}" "{'acc':"    
#     run_command_in_tmux "$tmux_session_3" "CUDA_VISIBLE_DEVICES=0 python main.py --entry pcc --model-path runs/dgcnn_curvenet_run_1_up_or_down_ratio_score_2_wrs/model_final.pth --exp-config configs/corruption/curvenet.yaml --use_upsample ${up} --sample_type ${down} --severity ${sev} --corruption ${cor}" "{'acc':"    

# done
# done

# # for cor in 'scale' 'jitter' 'dropout_global' 'dropout_local' 'add_global' 'add_local' 'rotate'; do 
# # # for cor in 'rotate'; do 
# # for sev in 0 1 2 3 4; do
# #     run_command_in_tmux "$tmux_session_4" "CUDA_VISIBLE_DEVICES=3 python main.py --entry oo3dc --model-path runs/dgcnn_curvenet_run_1_up_or_down_ratio_score_2_wrs/model_250.pth --exp-config configs/corruption/curvenet.yaml --use_upsample ${up} --sample_type ${down} --severity ${sev} --corruption ${cor}" "{'acc':"    
# #     run_command_in_tmux "$tmux_session_5" "CUDA_VISIBLE_DEVICES=3 python main.py --entry oo3dc --model-path runs/dgcnn_curvenet_run_1_up_or_down_ratio_score_2_wrs/model_275.pth --exp-config configs/corruption/curvenet.yaml --use_upsample ${up} --sample_type ${down} --severity ${sev} --corruption ${cor}" "{'acc':"    
# #     run_command_in_tmux "$tmux_session_6" "CUDA_VISIBLE_DEVICES=3 python main.py --entry oo3dc --model-path runs/dgcnn_curvenet_run_1_up_or_down_ratio_score_2_wrs/model_final.pth --exp-config configs/corruption/curvenet.yaml --use_upsample ${up} --sample_type ${down} --severity ${sev} --corruption ${cor}" "{'acc':"    
# # done
# # done

# for cor in 'occlusion' 'lidar' 'density_inc' 'density' 'cutout' 'uniform' 'gaussian' 'impulse' 'upsampling' 'background' 'rotation' 'shear' 'distortion' 'distortion_rbf' 'distortion_rbf_inv'; do
# for sev in 1 2 3 4 5; do
#     # run_command_in_tmux "$tmux_session_1" "CUDA_VISIBLE_DEVICES=3 python main.py --entry mnc --model-path runs/dgcnn_curvenet_run_1_up_or_down_ratio_score_2_wrs/model_250.pth --exp-config configs/corruption/curvenet.yaml --use_upsample ${up} --sample_type ${down} --severity ${sev} --corruption ${cor}" "{'acc':"    
#     # run_command_in_tmux "$tmux_session_2" "CUDA_VISIBLE_DEVICES=3 python main.py --entry mnc --model-path runs/dgcnn_curvenet_run_1_up_or_down_ratio_score_2_wrs/model_275.pth --exp-config configs/corruption/curvenet.yaml --use_upsample ${up} --sample_type ${down} --severity ${sev} --corruption ${cor}" "{'acc':"    
#     run_command_in_tmux "$tmux_session_3" "CUDA_VISIBLE_DEVICES=0 python main.py --entry mnc --model-path runs/dgcnn_curvenet_run_1_up_or_down_ratio_score_2_wrs/model_final.pth --exp-config configs/corruption/curvenet.yaml --use_upsample ${up} --sample_type ${down} --severity ${sev} --corruption ${cor}" "{'acc':"    
# done
# done
# done
# done

# for up in 'median_hroup'; do
# for down in 'wrs'; do
# # run_command_in_tmux "$tmux_session_3" "CUDA_VISIBLE_DEVICES=0 python main.py --entry test --model-path runs/dgcnn_curvenet_run_1_up_or_down_ratio_score_2_wrs/model_275.pth --exp-config configs/dgcnn_curvenet_run_1.yaml --use_upsample ${up} --sample_type ${down}" "{'acc':"    

# for cor in 'scale' 'jitter' 'dropout_global' 'dropout_local' 'add_global' 'add_local' 'rotate'; do 
# for sev in 0 1 2 3 4; do
#     run_command_in_tmux "$tmux_session_3" "CUDA_VISIBLE_DEVICES=0 python main.py --entry pcc --model-path runs/dgcnn_curvenet_run_1_up_or_down_ratio_score_2_wrs/model_final.pth --exp-config configs/corruption/curvenet.yaml --use_upsample ${up} --sample_type ${down} --severity ${sev} --corruption ${cor}" "{'acc':"    

# done
# done

# for cor in 'occlusion' 'lidar' 'density_inc' 'density' 'cutout' 'uniform' 'gaussian' 'impulse' 'upsampling' 'background' 'rotation' 'shear' 'distortion' 'distortion_rbf' 'distortion_rbf_inv'; do
# for sev in 1 2 3 4 5; do
#     # run_command_in_tmux "$tmux_session_1" "CUDA_VISIBLE_DEVICES=3 python main.py --entry mnc --model-path runs/dgcnn_curvenet_run_1_up_or_down_ratio_score_2_wrs/model_250.pth --exp-config configs/corruption/curvenet.yaml --use_upsample ${up} --sample_type ${down} --severity ${sev} --corruption ${cor}" "{'acc':"    
#     # run_command_in_tmux "$tmux_session_2" "CUDA_VISIBLE_DEVICES=3 python main.py --entry mnc --model-path runs/dgcnn_curvenet_run_1_up_or_down_ratio_score_2_wrs/model_275.pth --exp-config configs/corruption/curvenet.yaml --use_upsample ${up} --sample_type ${down} --severity ${sev} --corruption ${cor}" "{'acc':"    
#     run_command_in_tmux "$tmux_session_3" "CUDA_VISIBLE_DEVICES=0 python main.py --entry mnc --model-path runs/dgcnn_curvenet_run_1_up_or_down_ratio_score_2_wrs/model_final.pth --exp-config configs/corruption/curvenet.yaml --use_upsample ${up} --sample_type ${down} --severity ${sev} --corruption ${cor}" "{'acc':"    
# done
# done
# done
# done

# for up in 'median_hroup'; do
# for down in 'fps'; do
# for cor in 'scale' 'jitter' 'dropout_global' 'dropout_local' 'add_global' 'add_local' 'rotate'; do 
# for sev in 0 1 2 3 4; do
#     run_command_in_tmux "$tmux_session_1" "CUDA_VISIBLE_DEVICES=1 python main.py --entry pcc --model-path runs/dgcnn_curvenet_run_1_up_or_down_ratio_score_2_wrs/model_final.pth --exp-config configs/corruption/curvenet.yaml --use_upsample ${up} --sample_type ${down} --severity ${sev} --corruption ${cor}" "{'acc':"    
#     run_command_in_tmux "$tmux_session_2" "CUDA_VISIBLE_DEVICES=0 python main.py --entry pcc --model-path runs/dgcnn_pct_run_1_up_or_down_ratio_score_2_wrs/model_best_test.pth --exp-config configs/corruption/pct.yaml --use_upsample ${up} --sample_type ${down} --severity ${sev} --corruption ${cor}" "{'acc':"    

# done
# done
# for cor in 'scale' 'jitter' 'dropout_global' 'dropout_local' 'add_global' 'add_local' 'rotate'; do 
# for sev in 0 1 2 3 4; do
#     run_command_in_tmux "$tmux_session_1" "CUDA_VISIBLE_DEVICES=1 python main.py --entry oo3dc --model-path runs/dgcnn_curvenet_run_1_up_or_down_ratio_score_2_wrs/model_final.pth --exp-config configs/corruption/curvenet.yaml --use_upsample ${up} --sample_type ${down} --severity ${sev} --corruption ${cor}" "{'acc':"    
#     run_command_in_tmux "$tmux_session_2" "CUDA_VISIBLE_DEVICES=0 python main.py --entry oo3dc --model-path runs/dgcnn_pct_run_1_up_or_down_ratio_score_2_wrs/model_best_test.pth --exp-config configs/corruption/pct.yaml --use_upsample ${up} --sample_type ${down} --severity ${sev} --corruption ${cor}" "{'acc':"    
    
# done
# done

for cor in 'occlusion' 'lidar' 'density_inc' 'density' 'cutout' 'uniform' 'gaussian' 'impulse' 'upsampling' 'background' 'rotation' 'shear' 'distortion' 'distortion_rbf' 'distortion_rbf_inv'; do
for sev in 1 2 3 4 5; do
    # run_command_in_tmux "$tmux_session_5" "CUDA_VISIBLE_DEVICES=3 python main.py --entry mnc --model-path checkpoints/dgcnn_pointnet2_run_1_up_or_down_ratio_score_2_wrs_2/model_final.pth --exp-config configs/corruption/pointnet2.yaml --use_upsample median_hroup --sample_type ffps_0.95_2 --severity ${sev} --corruption ${cor}" "{'acc':"    
    # run_command_in_tmux "$tmux_session_5" "CUDA_VISIBLE_DEVICES=3 python main.py --entry mnc --model-path checkpoints/dgcnn_pointnet2_run_1_up_or_down_ratio_score_2_wrs_5/model_final.pth --exp-config configs/corruption/pointnet2.yaml --use_upsample median_hroup --sample_type ffps_0.95_5 --severity ${sev} --corruption ${cor}" "{'acc':"    
    # run_command_in_tmux "$tmux_session_5" "CUDA_VISIBLE_DEVICES=3 python main.py --entry mnc --model-path checkpoints/dgcnn_pointnet2_run_1_up_or_down_ratio_score_2_wrs_50/model_final.pth --exp-config configs/corruption/pointnet2.yaml --use_upsample median_hroup --sample_type ffps_0.95_50 --severity ${sev} --corruption ${cor}" "{'acc':"    
    # run_command_in_tmux "$tmux_session_5" "CUDA_VISIBLE_DEVICES=3 python main.py --entry mnc --model-path checkpoints/dgcnn_pointnet2_run_1_up_or_down_ratio_score_2_wrs_200/model_final.pth --exp-config configs/corruption/pointnet2.yaml --use_upsample median_hroup --sample_type ffps_0.95_200 --severity ${sev} --corruption ${cor}" "{'acc':"    
    run_command_in_tmux "$tmux_session_3" "CUDA_VISIBLE_DEVICES=2 python main.py --entry mnc --model-path checkpoints/dgcnn_pct_run_1_up_or_down_ratio_score_2_wrs_500/model_final.pth --exp-config configs/corruption/pct.yaml --use_upsample median_hroup --sample_type ffps_0.95_500 --severity ${sev} --corruption ${cor}" "{'acc':"    
    
done
done
# done
# done



# for up in 'median_hroup'; do
# for down in 'ffps_0.5' 'ffps_0.55' 'ffps_0.6' 'ffps_0.65' 'ffps_0.7' 'ffps_0.75' 'ffps_0.8' 'ffps_0.85' 'ffps_0.9'; do
#     run_command_in_tmux "$tmux_session_4" "CUDA_VISIBLE_DEVICES=0 python main.py --entry test --model-path runs/dgcnn_pct_run_1_up_or_down_ratio_score_2_wrs/model_best_test.pth --exp-config configs/dgcnn_pct_run_1.yaml --use_upsample ${up} --sample_type ${down}" "{'acc':"    
# for cor in 'scale' 'jitter' 'dropout_global' 'dropout_local' 'add_global' 'add_local' 'rotate'; do 
# for sev in 0 1 2 3 4; do
#     run_command_in_tmux "$tmux_session_4" "CUDA_VISIBLE_DEVICES=0 python main.py --entry pcc --model-path runs/dgcnn_pct_run_1_up_or_down_ratio_score_2_wrs/model_best_test.pth --exp-config configs/corruption/pct.yaml --use_upsample ${up} --sample_type ${down} --severity ${sev} --corruption ${cor}" "{'acc':"    

# done
# done

# for cor in 'occlusion' 'lidar' 'density_inc' 'density' 'cutout' 'uniform' 'gaussian' 'impulse' 'upsampling' 'background' 'rotation' 'shear' 'distortion' 'distortion_rbf' 'distortion_rbf_inv'; do
# for sev in 1 2 3 4 5; do
#     run_command_in_tmux "$tmux_session_4" "CUDA_VISIBLE_DEVICES=0 python main.py --entry mnc --model-path runs/dgcnn_pct_run_1_up_or_down_ratio_score_2_wrs/model_best_test.pth --exp-config configs/corruption/pct.yaml --use_upsample ${up} --sample_type ${down} --severity ${sev} --corruption ${cor}" "{'acc':"    
# done
# done
# done
# done


# for aug in 'rsmix' 'cutmix_r' 'cutmix_k' 'mixup'; do
# for up in 'no'; do
# # # for down in 'ffps_0.95'; do
# for down in 'fps'; do
#     run_command_in_tmux "$tmux_session_4" "CUDA_VISIBLE_DEVICES=0 python main.py --entry omni --model-path runs/${aug}_pct_run_1_no_fps/model_best_test.pth --exp-config configs/dgcnn_pct_run_1.yaml --use_upsample ${up} --sample_type ${down}" "{'acc':"    
#     run_command_in_tmux "$tmux_session_2" "CUDA_VISIBLE_DEVICES=3 python main.py --entry omni --model-path runs/${aug}_pct_run_1/model_best_test.pth --exp-config configs/dgcnn_pct_run_1.yaml --use_upsample ${up} --sample_type ${down}" "{'acc':"    
    
#     run_command_in_tmux "$tmux_session_3" "CUDA_VISIBLE_DEVICES=0 python main.py --entry omni --model-path runs/${aug}_pointnet2_run_1_no_fps/model_best_test.pth --exp-config configs/dgcnn_pointnet2_run_1.yaml --use_upsample ${up} --sample_type ${down}" "{'acc':"    

# done
# done
# done

# for up in 'no'; do
# # # for down in 'ffps_0.95'; do
# for down in 'no'; do
# for aug in 'rsmix' 'cutmix_r' 'cutmix_k' 'mixup' 'pgd';do 
#     run_command_in_tmux "$tmux_session_4" "CUDA_VISIBLE_DEVICES=3 python main.py --entry omni --model-path runs/${aug}_pointnet_run_1_no_no/model_best_test.pth --exp-config configs/dgcnn_pointnet_run_1.yaml --use_upsample ${up} --sample_type ${down}" "{'acc':"    
#     run_command_in_tmux "$tmux_session_2" "CUDA_VISIBLE_DEVICES=0 python main.py --entry omni --model-path runs/${aug}_pointnet_run_1/model_best_test.pth --exp-config configs/dgcnn_pointnet_run_1.yaml --use_upsample ${up} --sample_type ${down}" "{'acc':"    
#     run_command_in_tmux "$tmux_session_3" "CUDA_VISIBLE_DEVICES=3 python main.py --entry omni --model-path runs/${aug}_gdanet_run_1_no_no/model_best_test.pth --exp-config configs/dgcnn_gdanet_run_1.yaml --use_upsample ${up} --sample_type ${down}" "{'acc':"    


# done
# done
# done

# for up in 'no'; do
# for down in 'fps'; do
#     run_command_in_tmux "$tmux_session_4" "CUDA_VISIBLE_DEVICES=0 python main.py --entry omni --model-path runs/pgd_pct_run_1_no_fps/model_best_test.pth --exp-config configs/dgcnn_pct_run_1.yaml --use_upsample ${up} --sample_type ${down}" "{'acc':"    
# done
# done


# for model in 'pct' 'pointnet2'; do
#     for cor in 'scale' 'jitter' 'dropout_global' 'dropout_local' 'add_global' 'add_local' 'rotate'; do
#         for sev in 0 1 2 3 4; do
#             run_command_in_tmux "$tmux_session_1" "CUDA_VISIBLE_DEVICES=0 python main.py --entry pcc --model-path runs/rsmix_${model}_run_1/model_best_test.pth --exp-config configs/corruption/${model}.yaml --severity ${sev} --corruption ${cor}" "{'acc':"
#             run_command_in_tmux "$tmux_session_2" "CUDA_VISIBLE_DEVICES=0 python main.py --entry pcc --model-path runs/cutmix_r_${model}_run_1/model_best_test.pth --exp-config configs/corruption/${model}.yaml --severity ${sev} --corruption ${cor}" "{'acc':"
#             run_command_in_tmux "$tmux_session_3" "CUDA_VISIBLE_DEVICES=1 python main.py --entry pcc --model-path runs/cutmix_k_${model}_run_1/model_best_test.pth --exp-config configs/corruption/${model}.yaml --severity ${sev} --corruption ${cor}" "{'acc':"
#             run_command_in_tmux "$tmux_session_4" "CUDA_VISIBLE_DEVICES=1 python main.py --entry pcc --model-path runs/mixup_${model}_run_1/model_best_test.pth --exp-config configs/corruption/${model}.yaml --severity ${sev} --corruption ${cor}" "{'acc':"
#             run_command_in_tmux "$tmux_session_5" "CUDA_VISIBLE_DEVICES=1 python main.py --entry pcc --model-path runs/pgd_${model}_run_1/model_best_test.pth --exp-config configs/corruption/${model}.yaml --severity ${sev} --corruption ${cor}" "{'acc':"
#             run_command_in_tmux "$tmux_session_6" "CUDA_VISIBLE_DEVICES=2 python main.py --entry pcc --model-path runs/rsmix_${model}_run_1/model_best_test.pth --exp-config configs/corruption/${model}.yaml --severity ${sev} --corruption ${cor}" "{'acc':"
#             run_command_in_tmux "$tmux_session_7" "CUDA_VISIBLE_DEVICES=2 python main.py --entry pcc --model-path runs/cutmix_r_${model}_run_1/model_best_test.pth --exp-config configs/corruption/${model}.yaml --severity ${sev} --corruption ${cor}" "{'acc':"
#             run_command_in_tmux "$tmux_session_8" "CUDA_VISIBLE_DEVICES=3 python main.py --entry pcc --model-path runs/cutmix_k_${model}_run_1/model_best_test.pth --exp-config configs/corruption/${model}.yaml --severity ${sev} --corruption ${cor}" "{'acc':"
#             run_command_in_tmux "$tmux_session_9" "CUDA_VISIBLE_DEVICES=3 python main.py --entry pcc --model-path runs/mixup_${model}_run_1/model_best_test.pth --exp-config configs/corruption/${model}.yaml --severity ${sev} --corruption ${cor}" "{'acc':"
#         # run_command_in_tmux "$tmux_session_2" "python main.py --entry test --model-path runs/${aug}_${model}_run_1/model_best_test.pth --exp-config configs/dgcnn_${model}_run_1.yaml" "{'acc':"
#         done
#     done
# done


    # for aug in 'rsmix' 'cutmix_r' 'cutmix_k' 'mixup';do # 'pgd'; do