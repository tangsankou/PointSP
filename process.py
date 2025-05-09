import csv
import os
import re
def trans_txt_to_csv(base_path, folder, output,dataset,type):
    # 确保 base_path 和 output 都是文件夹路径
    if not os.path.isdir(base_path):
        raise NotADirectoryError("base_path does not exist or is not a directory")
    if not os.path.isdir(output):
        raise NotADirectoryError("output does not exist or is not a directory")

    # 遍历 folder 列表中的文件名
    for file in folder:
        # 构建完整的输入文件路径
        # file_name = file.split('/')[1]
        input_file_path = os.path.join(base_path, file + f'/{dataset}_{type}.txt')
        print("input_file_path:",input_file_path)
        # 构建完整的输出文件路径
        # output_file_path = os.path.join(output, file.split('/')[0])
        # os.makedirs(output_file_path, exist_ok=True)
        output_file = output+f"{file}_{type}.csv"
        print("output_file:",output_file)
        
        # 打开文本文件并读取数据
        with open(input_file_path, 'r') as file:
            lines = file.readlines()

        # 准备 CSV 文件的写入
        with open(output_file, 'w', newline='') as csvfile:
            # 创建 CSV 写入器
            csvwriter = csv.writer(csvfile, delimiter=',')
            
            # 写入表头
            header = ["Corruption", "Severity", "Acc", "Class Acc"]
            csvwriter.writerow(header)

            for line in lines:
                # 使用正则表达式匹配所需的部分
                match = re.search(r'Corruption: ([\w\s]+) Severity: (\d+) Acc: ([0-9.]+) Class Acc: ([\w\s]+)', line)
                if match:
                    corruption = match.group(1).strip()  # 移除两端的空白字符
                    severity = int(match.group(2))  # 将字符串转换为整数
                    acc = float(match.group(3))  # 将字符串转换为浮点数
                    class_acc = float(match.group(4))  # 将字符串转换为浮点数
                    # 写入 CSV 文件
                    csvwriter.writerow([corruption, severity, acc, class_acc])
                    # 如果需要将这些数据写入 CSV 或进行其他处理，可以在这里添加代码
                else:
                    print(f"No match found for line: {line}")


def calculate_er(csv_filename):
    cor = []  # 存储 Corruption 类别
    er_cor = []  # 存储每个 Corruption 类别计算出的 1-Acc 的平均值
    temp_er_sum = {}  # 临时存储相同 Corruption 类别的 er 值总和
    temp_er_count = {}  # 临时存储相同 Corruption 类别的 er 值数量

    # 打开 CSV 文件并读取数据
    with open(csv_filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            corruption = row['Corruption']
            acc = float(row['Acc'])
            er = 1 - acc
            
            # 累加相同 Corruption 类别的 er 值
            if corruption in temp_er_sum:
                temp_er_sum[corruption] += er
                temp_er_count[corruption] += 1
            else:
                temp_er_sum[corruption] = er
                temp_er_count[corruption] = 1
                cor.append(corruption)  # 添加新的 Corruption 类别

        # 计算每个 Corruption 类别的 1-Acc 平均值
        for corruption in temp_er_sum:
            er_cor.append(temp_er_sum[corruption] / temp_er_count[corruption])

    # 计算所有 Corruption 类别 1-Acc 平均值的平均值
    er = sum(er_cor) / len(er_cor) if er_cor else 0

    return cor, er_cor, er

def calculate_er_summary(folders, path, DATA_CORRUPTIONS,type):
    # 准备汇总 CSV 文件的写入
    with open(os.path.join(path, f'er_{type}.csv'), 'w', newline='') as csvfile:
        # 创建 CSV 写入器
        csvwriter = csv.writer(csvfile)
        # 写入表头
        header = ["file_name", "er"] + DATA_CORRUPTIONS
        csvwriter.writerow(header)
        
        # 遍历所有文件夹路径
        for folder in folders:
            file_name = folder+f"_{type}"
            folder_path = os.path.join(path, file_name+'.csv')
            cor, er_cor, er = calculate_er(folder_path)
            
            csvwriter.writerow([file_name, er] + [er_cor[DATA_CORRUPTIONS.index(corruption)] if corruption in DATA_CORRUPTIONS else 0 for corruption in cor])

def read_test(test_path,output,folders,type,model):
    # 读取文件夹中的 test.txt 文件，提取 Acc 并写入 CSV 文件
    output = output+f"{type}.csv"
    print("test:",output)
    with open(output, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # 写入表头
        csvwriter.writerow(['folder', 'acc'])

        for folder in folders:
            # 构建 test.txt 文件的完整路径
            txt_path = os.path.join(test_path, folder, f'test_{type}.txt')
            # 确保 test.txt 文件存在
            if os.path.isfile(txt_path):
                acc_sum = 0.0  # 用于累加 Acc 值
                acc_count = 0   # 用于计数 Acc 出现的次数
                with open(txt_path, 'r') as txtfile:
                    # 读取文件内容
                    for line in txtfile:
                        # 使用正则表达式匹配 'Acc:' 后面的浮点数
                        match = re.search(r'Acc: ([0-9.]+)', line)
                        if match:
                            acc_value = float(match.group(1))
                            acc_sum += acc_value
                            acc_count += 1
                # 计算均值
                if acc_count > 0:
                    average_acc = round(acc_sum / acc_count, 4)  # 保留四位小数
                    # 写入 CSV 文件
                    csvwriter.writerow([folder, average_acc])
            else:
                print(f"No such file or directory: '{txt_path}'")

import pandas as pd  
  
def merge_c(csv_files, output_path,first,other):  
    merged_data = pd.DataFrame()  
    first_file = True  
  
    for file_name in csv_files:  
        base_name = os.path.splitext(os.path.basename(file_name))[0]  
        print("new:",base_name)
        new_column_name = f'{base_name}'
        # 假设 file_name 是相对于某个目录的，您需要构建完整的文件路径  
        full_file_path = os.path.join(output_path, file_name)  # output_mnc 是外部定义的目录变量  
        print("full_file:",full_file_path)
        # 读取 CSV 文件  
        df = pd.read_csv(full_file_path, usecols=[first, other] if first_file else [other])  
  
        # 如果是第一个文件，将 file_name 列作为索引（如果需要的话）  
        if first_file:  
            df.rename(columns={other: new_column_name}, inplace=True)  
            merged_data = df[[first, new_column_name]]  # 只选择需要的两列  
            first_file = False 
        else:  
            # 将 er 列添加到合并后的 DataFrame 中，保留原有的索引（即 file_name）  
            df = df[[other]].rename(columns={other: new_column_name})  
            merged_data = pd.concat([merged_data, df], axis=1)  # 按列合并
    output_path = os.path.join(output_path, f'{other}.csv')  # 指定输出CSV文件的完整路径  
    merged_data.to_csv(output_path, index=False) 
  
    # 将合并后的数据写入新的 CSV 文件  
    merged_data.to_csv(output_path, index=False) 

def merge(csv_files, output_path,first,other,model):  
    merged_data = pd.DataFrame()  
    first_file = True  
  
    for file_name in csv_files:  
        base_name = os.path.splitext(os.path.basename(file_name))[0]  
        print("base_name:",base_name)
        new_column_name = f'{other}_{base_name}'
        # 假设 file_name 是相对于某个目录的，您需要构建完整的文件路径  
        full_file_path = os.path.join(output_path, file_name)  # output_mnc 是外部定义的目录变量  
        print("full_file:",full_file_path)
        # 读取 CSV 文件  
        df = pd.read_csv(full_file_path, usecols=[first, other] if first_file else [other])  
  
        # 如果是第一个文件，将 file_name 列作为索引（如果需要的话）  
        if first_file:  
            df.rename(columns={other: new_column_name}, inplace=True)  
            merged_data = df[[first, new_column_name]]  # 只选择需要的两列  
            first_file = False 
        else:  
            # 将 er 列添加到合并后的 DataFrame 中，保留原有的索引（即 file_name）  
            df = df[[other]].rename(columns={other: new_column_name})  
            merged_data = pd.concat([merged_data, df], axis=1)  # 按列合并
    output_path = output_path +f"test-{model}"
    output_path = os.path.join(output_path, f'{other}.csv')  # 指定输出CSV文件的完整路径  
    merged_data.to_csv(output_path, index=False) 
  
    # 将合并后的数据写入新的 CSV 文件  
    merged_data.to_csv(output_path, index=False) 

if __name__=='__main__':
    base_path="/home/user_tp/workspace/code/attack/ModelNet40-C/runs/"
    # folders_all = ['dgcnn_pct_run_1_no_fps','dgcnn_pct_run_1_no_wfps','dgcnn_pct_run_1_no_wrs','dgcnn_pct_run_1_r_rwup_fps','dgcnn_pct_run_1_r_rwup_wfps','dgcnn_pct_run_1_r_rwup_wrs',
    #            'dgcnn_pct_run_1_up_or_down_fps','dgcnn_pct_run_1_up_or_down_wfps','dgcnn_pct_run_1_up_or_down_wrs',
    #            'dgcnn_pointnet2_run_1_no_fps','dgcnn_pointnet2_run_1_no_wfps','dgcnn_pointnet2_run_1_no_wrs','dgcnn_pointnet2_run_1_r_rwup_fps','dgcnn_pointnet2_run_1_r_rwup_wfps',
    #            'dgcnn_pointnet2_run_1_r_rwup_wrs','dgcnn_pointnet2_run_1_up_or_down_fps','dgcnn_pointnet2_run_1_up_or_down_wfps','dgcnn_pointnet2_run_1_up_or_down_wrs',
    #            'cutmix_k_pct_run_1','cutmix_r_pct_run_1','mixup_pct_run_1','pgd_pct_run_1','rsmix_pct_run_1',
    #            'cutmix_k_pointnet2_run_1','cutmix_r_pointnet2_run_1','mixup_pointnet2_run_1','rsmix_pointnet2_run_1']
    
    # augs =['cutmix_k','cutmix_r','mixup','rsmix']
    # models = ['pct','pointnet2']

    # ###cal er
    MNC_CORRUPTIONS = ["occlusion", "lidar", "density_inc", "density", "cutout", "uniform", "gaussian", "impulse", "upsampling", "background", "rotation", "shear", "distortion", "distortion_rbf", "distortion_rbf_inv"]  
    PCC_CORRUPTIONS = ['scale','jitter','dropout_global','dropout_local','add_global','add_local','rotate']
    dataset='mnc'
    model_c = 'pct_down'
    output_mnc=f"/home/user_tp/workspace/code/attack/ModelNet40-C/runs/mnc-{model_c}/"
    output_pcc=f"/home/user_tp/workspace/code/attack/ModelNet40-C/runs/pcc-{model_c}/"
    os.makedirs(output_mnc, exist_ok=True)
    os.makedirs(output_pcc, exist_ok=True)

    folders_pct = ['dgcnn_pct_run_1_no_fps','dgcnn_pct_run_1_no_wfps','dgcnn_pct_run_1_no_wrs','dgcnn_pct_run_1_r_rwup_fps','dgcnn_pct_run_1_r_rwup_wfps','dgcnn_pct_run_1_r_rwup_wrs',
               'dgcnn_pct_run_1_up_or_down_fps','dgcnn_pct_run_1_up_or_down_wfps','dgcnn_pct_run_1_up_or_down_wrs']
    folders_pn2 = ['dgcnn_pointnet2_run_1_no_fps','dgcnn_pointnet2_run_1_no_wfps','dgcnn_pointnet2_run_1_no_wrs','dgcnn_pointnet2_run_1_r_rwup_fps','dgcnn_pointnet2_run_1_r_rwup_wfps','dgcnn_pointnet2_run_1_r_rwup_wrs',
               'dgcnn_pointnet2_run_1_up_or_down_fps','dgcnn_pointnet2_run_1_up_or_down_wfps','dgcnn_pointnet2_run_1_up_or_down_wrs']
    csv_files = []
    for up in ['no','r_rwup']:
        for down in ['fps','wfps','wrs']:
            type = f"{up}_{down}"
            csv_files.append(f"{dataset}_{type}/er_{type}.csv")
    print(csv_files)
    for up in ['up_or_down_0.1','up_or_down_0.2','up_or_kdown_0.1','up_or_kdown_0.2']:
        for down in ['wrs']:
    # for up in ['no','r_rwup']:
    #     for down in ['fps','wfps','wrs']:
            type = f"{up}_{down}"
            output =output_mnc+f'{dataset}_{type}/'
            os.makedirs(output, exist_ok=True)
            trans_txt_to_csv(base_path,folders_pn2,output,dataset,type)
            calculate_er_summary(folders_pn2,output,MNC_CORRUPTIONS,type)
    # merge_c(csv_files,output_mnc,"file_name","er")


    ###merge test
    # output_test =f"/home/user_tp/workspace/code/attack/ModelNet40-C/runs/"
    # model = 'pointnet2'
    # folder=[]
    # output = output_test+f"test-{model}-down/"
    # os.makedirs(output, exist_ok=True)

    # for up in ['up_or_down_0.1','up_or_down_0.2','up_or_kdown_0.1','up_or_kdown_0.2']:
    #     for down in ['wrs']:
    #         type = f"{up}_{down}"
    #         folder.append(f"dgcnn_{model}_run_1_{type}")
    # print("fold:",folder)
    # test_files=[]
    # for up in ['no','r_rwup']:
    #     for down in ['wfps']:
    #         type = f"{up}_{down}"
    #         test_files.append(f"test-{model}-down/{type}.csv")
    #         read_test(output_test,output,folder,type,model)

    # # print("test_files:",test_files)
    # merge(test_files,output_test,"folder","acc",model)

###我想要merge的输入是什么？他们的文件结构不一样，我希望输入一个文件地址库，一个输出地址，然后first和other，
    

    ###我想改成一体化的过程，在循环的时候不仅把文件读取和计算的工作给做了，还要把文件目录生成，以便我合并那些文件。
    ###但是现在有一个问题是，不知道pct和pointnet2的要不要合并处理，那样的话可能要考虑不仅在df里面加列，还要加行。我觉得还是处理一个比较好。
    # folder_da = ["pgd_pct_run_1"]
    # for model in models:
    #     for aug in augs:
    #         folder_da.append(f"{aug}_{model}_run_1")
    # print("da:",folder_da)
    
    # folder=[]
    # for model in models:
    #     for up in ['no','r_rwup','up_or_down']:
    #         for down in ['fps','wfps','wrs']:
    #             type = f"{up}_{down}"
    #             output =output_mnc+f'{dataset}_{type}/'
    #             folder.append(f"dgcnn_{model}_run_1_{type}")
    #             read_test(base_path,folders,type)
    #             trans_txt_to_csv(base_path,folders,output,dataset,type)
    #             # cor, er_cor, er=calculate_er(output)
    #             calculate_er_summary(folders,output,DATA_CORRUPTIONS,type)
    # print("folder:",folder)
    # folder_with_csv = [f'{item}.csv' if not item.endswith('.csv') else item for item in folder]  
    # print(folder_with_csv)