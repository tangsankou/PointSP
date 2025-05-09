import os
import csv
import pandas as pd
import re
import matplotlib.pyplot as plt

#for wrs' k of knn
def extract_folder_numbers(folder_path):
    target_folders = [folder for folder in os.listdir(folder_path) if folder.startswith("m")]# mwrs_k_
    numbers = [int(re.search(r'^m(\d+)', folder).group(1)) for folder in target_folders]
    return numbers

def read_eval(eval_path):
    """#一次的测试结果 
    with open(eval_path, 'r') as eval_file:
    # 读取文件内容
        content = eval_file.read()
        # 查找Test Instance Accuracy和Class Accuracy字段
        test_instance_accuracy_start = content.find("Test Instance Accuracy:") + len("Test Instance Accuracy:")
        test_instance_accuracy_end = content.find(",", test_instance_accuracy_start)
        test_instance_accuracy = float(content[test_instance_accuracy_start:test_instance_accuracy_end].strip())

        class_accuracy_start = content.find("Class Accuracy:") + len("Class Accuracy:")
        class_accuracy_end = content.find("\n", class_accuracy_start)
        class_accuracy = float(content[class_accuracy_start:class_accuracy_end].strip())
            # 写入csv文件 """
    test_instance_accuracies = []
    class_accuracies = []
    with open(eval_path, 'r') as eval_file:
        content = eval_file.read()
        # 使用正则表达式匹配测试结果的模式
        pattern = re.compile(r"Acc: (\d+\.\d+), Class Acc: (\d+\.\d+)")
        matches = re.finditer(pattern, content)
        for match in matches:
            # 从每个匹配中提取测试准确度
            test_instance_accuracy = float(match.group(1))
            class_accuracy = float(match.group(2))
            # 将测试准确度存储到列表中
            test_instance_accuracies.append(test_instance_accuracy)
            class_accuracies.append(class_accuracy)
    # 计算平均值
    instance_accuracy = sum(test_instance_accuracies) / len(test_instance_accuracies) if test_instance_accuracies else 0
    class_accuracy = sum(class_accuracies) / len(class_accuracies) if class_accuracies else 0
    return instance_accuracy,class_accuracy

def calculate_mce(csv_path):
    df = pd.read_csv(csv_path, header=None)
    ce_values = []
    cor = []
    count = 1
    instance_acc = float(df.iloc[1, 1])
    accum_ce = (1 - instance_acc) #er
    # accum_ce = (1 - instance_acc) / (1 - float(dgcnn.iloc[1, 1])) #mce
    temp = df.iloc[1, 0]
    for i in range(2, len(df)):
        cor_instance_acc = float(df.iloc[i, 1])  # 将字符串转换为浮点数       
        name = temp.rsplit('_', 1)[0]
        # print("name:",name)
        cei =  (1 - cor_instance_acc)#er
        # cei = (1 - cor_instance_acc) / (1 - float(dgcnn.iloc[i, 1])) # mce
        if "_".join(df.iloc[i, 0].split("_")[:-1]) == "_".join(temp.split("_")[:-1]):
            count += 1
            accum_ce += cei
        else:
            ce = accum_ce / count
            ce_values.append(ce)# 将结果添加到列表
            cor.append(name)
            temp = df.iloc[i, 0]
            accum_ce = cei 
            count = 1          
    # print("ce_values:",ce_values)
    # 计算所有的均值 mCE
    mce = sum(ce_values) / len(ce_values)
    return cor, ce_values, mce

def cal_mce(path, datype, data_clean):
    csv_files = [file for file in os.listdir(path) if file.endswith('.csv')]# and file.startswith('m')]#mwrs_k_
    # 初始化一个 DataFrame 用于保存结果
    result_df = pd.DataFrame()
    # print("csv_files",csv_files)
    # 初始化一个列表用于保存 cor 数据
    cor_data = []
    path_dgcnn = "/home/user_tp/workspace/code/base_model/PCT_Pytorch-main/checkpoints/dgcnn/"
    if datype == "mnc":
        dpath = path_dgcnn + "modelnet40c/dgcnn.csv"
    elif datype == "pcc":
        dpath = path_dgcnn + "pointcloudc/dgcnn.csv"
    dgcnn = pd.read_csv(dpath, header=None)
    for csv_file in csv_files:
        file_name = csv_file.split(".csv")[0]
        print("file:",file_name)
        csv_path = os.path.join(path, csv_file)
        er_path = path+'/ER'
        if not os.path.exists(er_path):
            os.makedirs(er_path)
        #提取数字k
        # k = file_name.split("_")[2]#"m_wrs_0.1"->"0.1"
            # print("k:",k)
        # cor, ce, mce = calculate_mce(csv_path, dgcnn, data_clean)
        cor, ce, mce = calculate_mce(csv_path)
        
        print("cor ce mce:",cor,ce,mce)
        # 如果 cor_data 为空，将 cor 数据添加进去
        if not cor_data:
            cor_data.extend(['er'])#er
            cor_data.extend(cor)
            # cor_data.extend(['ce'])#mce
        # ce = mce + ce
        ce.insert(0, mce)
        # ce.append(mce)
        result_df[file_name] = ce# 将 ce 数据添加到 result_df
    # result_df_sorted = result_df.sort_index(axis=1)
    result_df.insert(0, 'file_name', cor_data)# 将 cor_data 添加到 result_df 的第一列
    result_df_t = result_df.T
    print("----------",result_df_t)
    result_df_t.to_csv(os.path.join(er_path, f"er.csv"))#er
    # result_df_t.to_csv(os.path.join(path, f"mce.csv"), index=False)#mce

def read_modelnet40c(base_path, folder_name, output_path):
    # 读取modelnet40c.txt文件
    data_clean = 'original'
    fold_path = os.path.join(base_path, folder_name, "mnc.txt")
    # output_path = base_path + f"modelnet40c/{mnc}"
    if os.path.exists(fold_path):
        with open(fold_path, 'r') as modelnet40c_file:
            content = modelnet40c_file.read()
            # 找到"Test Instance Accuracy:"和"Class Accuracy:"字段的起始位置
            test_instance_accuracy_start = content.find("Acc:")
            class_accuracy_start = content.find("Class Acc:")
            # 确保找到这两个字段
            if test_instance_accuracy_start != -1 and class_accuracy_start != -1:
                # 获取字段所在行的位置
                test_instance_accuracy_line_start = content.rfind("\n", 0, test_instance_accuracy_start) + 1
                class_accuracy_line_start = content.rfind("\n", 0, class_accuracy_start) + 1
                # 截取包含字段的部分
                relevant_content = content[max(test_instance_accuracy_line_start, class_accuracy_line_start):]
                # 找到"Test Instance Accuracy:"和"Class Accuracy:"字段的行数
                data_uniform_line_start = relevant_content.find("occlusion")
                data_uniform_line_start = relevant_content.rfind("\n", 0, data_uniform_line_start) + 1
                relevant_content = relevant_content[data_uniform_line_start:]# 截取包含字段的部分
                rows = relevant_content.split("\n")
                csv_file = os.path.join(output_path, folder_name.split('/')[0] + '/')
                os.makedirs(csv_file, exist_ok=True)

                csv_file_path = os.path.join(output_path, f"{folder_name}.csv")

                with open(csv_file_path, mode='w', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(["corruption", "Acc", "Class Acc"])
                    # 遍历每行数据
                    for row in rows:
                        data_start = row.find("Corruption: occlusion Severity: 1")
                        if data_start != -1:
                            data_end = row.find(":", data_start)
                            data = row[data_start:data_end].strip()

                            test_instance_accuracy_start = row.find("Acc:") + len("Acc:")
                            test_instance_accuracy_end = row.find(" ", test_instance_accuracy_start)
                            test_instance_accuracy = float(row[test_instance_accuracy_start:test_instance_accuracy_end].strip())

                            class_accuracy_start = row.find("Class Acc:") + len("Class Acc:")
                            class_accuracy_end = row.find(" ", class_accuracy_start)
                            class_accuracy = float(row[class_accuracy_start:class_accuracy_end].strip())
                            csv_writer.writerow([data, test_instance_accuracy, class_accuracy])
                # print(f"结果已写入 {csv_file_path}")
            else:
                print(f"字段未找到在文件夹: {folder_name}")
    else:
        print(f"mnc.txt文件未找到在文件夹: {folder_name}") 
    ###calculate mce
    datype = "mnc"
    cal_mce(output_path, datype, data_clean)

def read_pointcloudc(base_path, folder_name,output_path):
    data_clean = 'clean'
    # 读取pointcloudc.txt文件
    fold_path = os.path.join(base_path, folder_name, f"pointcloudc.txt")
    # output_path = base_path + f"pointcloudc/{pcc}"
    if os.path.exists(fold_path):
        with open(fold_path, 'r') as pointcloudc_file:
            content = pointcloudc_file.read()
            # 找到"Test Instance Accuracy:"和"Class Accuracy:"字段的起始位置
            test_instance_accuracy_start = content.find("Test Instance Accuracy:")
            class_accuracy_start = content.find("Class Accuracy:")
            # 确保找到这两个字段
            if test_instance_accuracy_start != -1 and class_accuracy_start != -1:
                # 获取字段所在行的位置
                test_instance_accuracy_line_start = content.rfind("\n", 0, test_instance_accuracy_start) + 1
                class_accuracy_line_start = content.rfind("\n", 0, class_accuracy_start) + 1
                # 截取包含字段的部分
                relevant_content = content[max(test_instance_accuracy_line_start, class_accuracy_line_start):]
                # 找到"Test Instance Accuracy:"和"Class Accuracy:"字段的行数
                data_uniform_line_start = relevant_content.find("data_scale_0")
                data_uniform_line_start = relevant_content.rfind("\n", 0, data_uniform_line_start) + 1
                relevant_content = relevant_content[data_uniform_line_start:]# 截取包含字段的部分
                rows = relevant_content.split("\n")
                csv_file_path = os.path.join(output_path, f"{folder_name}.csv")
                with open(csv_file_path, mode='w', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(["corruption", "Test Instance Accuracy", "Class Accuracy"])
                    # 遍历每行数据
                    for row in rows:
                        data_start = row.find("data_")
                        if data_start != -1:
                            # 提取字段数据
                            data_end = row.find(":", data_start)
                            data = row[data_start:data_end].strip()

                            test_instance_accuracy_start = row.find("Test Instance Accuracy:") + len("Test Instance Accuracy:")
                            test_instance_accuracy_end = row.find(",", test_instance_accuracy_start)
                            test_instance_accuracy = float(row[test_instance_accuracy_start:test_instance_accuracy_end].strip())

                            class_accuracy_start = row.find("Class Accuracy:") + len("Class Accuracy:")
                            class_accuracy_end = row.find(",", class_accuracy_start)
                            class_accuracy = float(row[class_accuracy_start:class_accuracy_end].strip())
                            csv_writer.writerow([data, test_instance_accuracy, class_accuracy])
                # print(f"结果已写入 {csv_file_path}")
            else:
                print(f"字段未找到在文件夹: {folder_name}")
    else:
        print(f"pointcloudc.txt文件未找到在文件夹: {folder_name}") 
    ###calculate mce
    datype = "pcc"
    cal_mce(output_path, datype, data_clean)

""" def draw(file_path, save_path, i=1, j=-1):
    df = pd.read_csv(file_path)# 读取 Excel 表格数据
    x_data = df.columns[1:]# 获取第一行作为 x 轴数据
    # print("x_data:",x_data)
    # x_data = list(map(float, df.columns[1:]))
    # x_data = df.columns[1:].map(lambda x: int(x) if x in df.columns[1:] else x)
    # x_data = df.columns[1:].astype(int)
    y_data = df.iloc[j, 1:]# 获取最后一行作为 y 轴数据
    # print("y_data:",y_data)
    x_label = df.columns[0][0]
    y_label = df.iloc[j, 0]
    # print("x_label:",x_label)
    # print("y_label:",y_label)
    title = f"Line Chart for {y_label} of different k"
    plt.figure()  # 创建新的 Figure 对象
    plt.plot(x_data, y_data, marker='o')# 生成折线图
    # 添加标题和标签
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label) # 第j行第0个数据作为y轴标签
    # plt.legend()
    plt.savefig(save_path)
    plt.show() """

def draw(file_path, save_path,fps, i=1, j=-1):
    df = pd.read_csv(file_path)# 读取 Excel 表格数据
    x_data = df.columns[1:]
    y_data = df.iloc[j, 1:]
    x_label = df.columns[0][0]
    y_label = df.iloc[j, 0]
    title = f"Line Chart for {y_label} of different k"
    plt.figure(figsize=(8, 6))
    plt.plot(x_data, y_data, color='b', label='wrs', marker='o')
    plt.hlines(y=fps, xmin=-1, xmax=max(x_data), color='r', linestyle='--', label='fps')
    # 在 y 轴上标注 fps 的值
    plt.text(max(x_data), fps, f'{round(fps, 4)}', color='r',verticalalignment='center')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # 调整右上角标签的位置
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
    # 设置标题
    plt.title(title)
    plt.savefig(save_path)
    plt.show()

#cal and save the er or mce of different k
def save_by_k():
    base_path = "/home/user_tp/workspace/code/base_model/PCT_Pytorch-main/checkpoints/wrs/"
    # 获取所有文件夹并按名称排序
    # folders = sorted(os.listdir(base_path))
    folders = [folder for folder in os.listdir(base_path) if folder.startswith("m")]
    # 创建modelnet40c文件夹用于存放csv文件
    mnc = "modelnet40c"
    pcc = "pointcloudc"

    os.makedirs(os.path.join(base_path, mnc), exist_ok=True)
    os.makedirs(os.path.join(base_path, pcc), exist_ok=True)
    test_df = pd.DataFrame(columns=["k", "instance_acc", "class_acc"])# 创建一个空的 DataFrame
    new_rows = []    # 存储 new_row 的列表
    for folder_name in folders:
        folder_path = os.path.join(base_path, folder_name) 
        if os.path.isdir(folder_path):# 检查是否为文件夹
            eval_file = os.path.join(folder_path, "eval.txt")
            match = re.search(r'^m(\d+)', folder_name)# 使用正则表达式提取数字部分
            if match is not None:
                k = int(match.group(1))
                # k = match.group(1)
                if os.path.exists(eval_file):
                    instance_acc, class_acc = read_eval(eval_file)
                    new_row = pd.DataFrame({"k": [k], "instance_acc": [instance_acc], "class_acc": [class_acc]})
                    new_rows.append(new_row)
                read_modelnet40c(base_path, folder_name, mnc)
                read_pointcloudc(base_path, folder_name,pcc)
    test_df = pd.concat([test_df] + new_rows, ignore_index=True)# 将所有 new_row 合并到 test_df
    test_df.sort_values(by="k", inplace=True)#排序
    test_df_t = test_df.T#转置
    test_df_t.to_csv(os.path.join(base_path, f"wrs_test.csv"), header=False, index=True)
    """folder_path = "/home/user_tp/workspace/code/base_model/Pointnet_Pointnet2_pytorch/log/class_pointnet2"
    folder_numbers = extract_folder_numbers(folder_path)
    print("Extracted numbers:", folder_numbers)""" 
    ###draw image
    # file_path1 = base_path + "modelnet40c/" +"mce.csv"
    # save_path1 = base_path + "modelnet40c/" +"mce.png"
    # draw(file_path1, save_path1,0.8377108444238959, 1, -1)
    file_path1 = base_path + f"{mnc}/" +"er.csv"#er
    save_path1 = base_path + f"{mnc}/" +"er.png"#er
    draw(file_path1, save_path1,0.20226902666666663, 1, -1)#er
    file_path2 = base_path + f"{pcc}/" +"er.csv"#er
    save_path2 = base_path + f"{pcc}/" +"er.png"#er
    draw(file_path2, save_path2,0.219, 1, -1)#er
    # file_path2 = base_path + f"{pcc}/" +"mce.csv"
    # save_path2 = base_path + f"{pcc}/" +"mce.png"
    # draw(file_path2, save_path2,1.0708467187039232, 1, -1)
    file_path3 = base_path +"wrs_test.csv"    
    save_path3 = base_path +"wrs_test.png"
    draw(file_path3, save_path3,0.929092,1, 0)

#cal and save the er or mce of different file
def save_by_file(base_path):
    # folders = [folder for folder in os.listdir(base_path)]
    # folders = ['pn2_ssg_roup_wfps']
    folders = ['rnum/pct_fps','rnum/pct_wfps_t20','up/pct_oup_fps','up/pct_roup_fps','up/pct_rrwup_fps',
               'up/pct_rwup_fps','up_wfps/pct_oup_wfps_t20','up_wfps/pct_roup_wfps_t20']
    # 创建modelnet40c文件夹用于存放csv文件
    # folders = ['fps']
    type = '8'
    mnc = f"modelnet40c-{type}"
    # pcc = f"pointcloudc-{type}"
    path_mnc = base_path+f'modelnet40c/{mnc}'
    # path_pcc = base_path+f'pointcloudc/{pcc}'
    os.makedirs(path_mnc, exist_ok=True)
    # os.makedirs(path_pcc, exist_ok=True)
    test_df = pd.DataFrame(columns=["file", "acc", "class acc"])# 创建一个空的 DataFrame
    new_rows = []    # 存储 new_row 的列表
    for folder_name in folders:
        folder_path = os.path.join(base_path, folder_name) 
        if os.path.isdir(folder_path):# 检查是否为文件夹
            eval_file = os.path.join(folder_path, "test.txt")
            # k = match.group(1)
            if os.path.exists(eval_file):
                instance_acc, class_acc = read_eval(eval_file)
                new_row = pd.DataFrame({"file": [folder_name], "acc": [instance_acc], "class acc": [class_acc]})
                new_rows.append(new_row)
            read_modelnet40c(base_path, folder_name,path_mnc)
            # read_pointcloudc(base_path, folder_name,path_pcc)
    test_df = pd.concat([test_df] + new_rows, ignore_index=True)# 将所有 new_row 合并到 test_df
    # test_df_t = test_df.T#转置
    test_df.to_csv(os.path.join(base_path, f"test/test_{type}.csv"), header=False, index=True)
    

if __name__=='__main__':
    # 指定路径
    base_path = "/home/user_tp/workspace/code/attack/ModelNet40-C/cor_exp/modify/"
    save_by_file(base_path)

    #dgcnn
    # pathd = "/home/user_tp/workspace/code/base_model/PCT_Pytorch-main/checkpoints/dgcnn/"
    """ output_folder_path1 = pathd + "modelnet40c"
    modelnet40c_file = output_folder_path1 + ".txt"
    fold_name = "dgcnn"
    output_folder_path2 = pathd + "pointcloudc"
    pointcloudc_file = output_folder_path2 + ".txt"
    read_modelnet40c(output_folder_path1, fold_name)
    read_pointcloudc(output_folder_path2, fold_name) """
    