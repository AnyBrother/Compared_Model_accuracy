# Compared_Model_accuracy
This code is for calculating and output the predicted probability/labels/accuracy of classical models.

# Purpose

(**In Chinese**) 实现从excel(第一行为列名,最后一列是标签列, 且excel文件与该python代码处于同一个文件夹内)读取数据, 计算指定模型的精度,并将精度结果输出成excel文件.

(**In English**) Enable reading data from `excel` data document (First row is `name of index`, last column is `labels 0/1`), which is located in the same file with this python code, auto-calculate accuracy of desired models and output results into 'excel'.

# How to use it
(**In Chinese**) 将读取的数据文件和该python代码放在同一个文件夹内. 并修改`main()`函数中的496-502行参数, 之后直接整体运行即可.

(**In English**) Put the data and code in the same file folder. Then the only thing you need to do is to modify the paramaters located from row 496 to row 502 in function `main()`.

```Python
    excel_file_name_train = "Data_in.xlsx"  # 可替换成自己想读取的-训练-数据文件名+后缀  The input train data file name
    excel_sheet_name_train = "Sheet_train"  # 可替换成自己想读取的-训练-数据文件中的sheet名  The input train data sheet name
    excel_file_name_test = "Data_in.xlsx"  # 可替换成自己想读取的-测试-数据文件名+后缀  The input test data file name
    excel_sheet_name_test = "Sheet_test"  # 可替换成自己想读取的-测试-数据文件中的sheet名   The input test data sheet name
    model_list = ["LG", "RF"]  # 可替换成自己想使用的模型名称, 名称缩写需从BasePredictYkp()已有的选择   The input model abbreviation from 'BasePredictYkp()'
    output_excel_file_name = "Data_out.xlsx"  # 可自定义为想输出的excel文件名+后缀  The output excel name which could be self-define.
```

# What is the output like
**Output 1**: `2020.04.04.11：54.Data_out.xlsx`, which is an `excel` file with the predicted probability/labels/accuracy of train & test dataset, by models trained with train-dataset.   

**Output 2**: `2020.04.04.11：54.保存的模型所有变量结果.p`, which is an `pickle` file with all the results, including the trained models. So as to check and help invocation model
