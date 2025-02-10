import pandas as pd

# 读取Excel文件
df = pd.read_excel("merged_result.xlsx")

# 显示前几行
print("DataFrame 信息:")
print(df.info())
print("\n前5行数据:")
print(df.head()) 