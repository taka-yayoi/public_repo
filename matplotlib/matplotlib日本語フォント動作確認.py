# Databricks notebook source
# MAGIC %md # matplotlibにおける日本語フォントの動作確認
# MAGIC <br>
# MAGIC - 事前に別添の `init scriptによる日本語フォントのインストール` を参考にinit scriptを設定してください。

# COMMAND ----------

# init script動作確認
#display(dbutils.fs.ls("dbfs:/cluster-logs/0215-005452-lump676/init_scripts/0215-005452-lump676_10_0_1_129/"))
#dbutils.fs.head("dbfs:/cluster-logs/0215-005452-lump676/init_scripts/0215-005452-lump676_10_0_1_129/20210221_015534_00_japanese-font-install.sh.stderr.log")
#dbutils.fs.head("dbfs:/cluster-logs/0215-005452-lump676/init_scripts/0215-005452-lump676_10_0_1_129/20210221_015534_00_japanese-font-install.sh.stdout.log")
#dbutils.fs.rm("dbfs:/cluster-logs/", recurse=True)

# COMMAND ----------

import matplotlib 
import matplotlib.font_manager as fm 
originalFilelist = fm.findSystemFonts()

import numpy as np
import matplotlib.pyplot as plt

# COMMAND ----------

# インストール済みフォントの表示
originalFilelist

# COMMAND ----------

# 日本語フォントの存在を確認
fm.findfont('TakaoGothic')

# COMMAND ----------

# MAGIC %md ## 日本語フォントのテスト

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC 
# MAGIC plt.ion()
# MAGIC fig2 = plt.figure()
# MAGIC axl = fig2.add_subplot(1,1,1)
# MAGIC data1 = np.arange(1001)
# MAGIC matplotlib.rc('font', family='TakaoGothic')
# MAGIC line1, = axl.plot(data1[:], label="あああtest Value")
# MAGIC axl.legend(loc="best")
# MAGIC csfont = {'fontname':'TakaoPMincho'}
# MAGIC hfont = {'fontname':'TakaoPMincho'}
# MAGIC display(fig2)

# COMMAND ----------

# MAGIC %md ## 日本語フォントの比較

# COMMAND ----------

#that's from http://olsgaard.dk/showing-japanese-characters-in-matplotlib-on-ubuntu.html

fig = plt.figure()

fonts = ['TakaoPMincho', 'TakaoMincho', 'TakaoExMincho', 'TakaoGothic', 'DejaVu Sans']
#fonts = ['Arial', 'Times New Roman', 'Helvetica'] #uncomment this line on Windows and see if it helps!
english = 'The quick ...'
japanese = '日本語'
x = 0.1
y = 1

# Buils headline
plt.text(x+0.5,y, 'english')
plt.text(x+0.7, y, 'japanese')
plt.text(x,y, 'Font name')
plt.text(0,y-0.05, '-'*100)
y -=0.1

for f in fonts:
    matplotlib.rc('font', family='DejaVu Sans')
    plt.text(x,y, f+':')
    matplotlib.rc('font', family=f)
    plt.text(x+0.5,y, english)
    plt.text(x+0.7, y, japanese)
    y -= 0.1
    print(f, fm.findfont(f))  # Sanity check. Prints the location of the font. If the font it not found, an error message is printed and the location of the fallback font is shown

display(fig)
