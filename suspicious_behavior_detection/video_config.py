# Databricks notebook source
# DBTITLE 1,画像を抽出する動画、抽出画像の保存場所の設定
# ソースのトレーニング動画
srcVideoPath = "/databricks-datasets/cctvVideos/train/"

# ソースのテスト動画
srcTestVideoPath = "/databricks-datasets/cctvVideos/test/"

# ソースのトレーニング動画(MP4)
srcVideoMP4Path = "/databricks-datasets/cctvVideos/mp4/train/"

# ソースのテスト動画(MP4)
srcTestVideoMP4Path = "/databricks-datasets/cctvVideos/mp4/test/"

# CCTV動画のラベルデータ
labeledDataPath = "/databricks-datasets/cctvVideos/labels/"


#
# 以下のディレクトリパスwork_pathを設定してください
#
work_path = "/tmp/takaakiyayoidatabrickscom"

# (動画からの)抽出トレーニング画像格納パス
targetImgPath = f"{work_path}/videos/cctvFrames/train/"

# (動画からの)抽出テスト画像格納パス
targetImgTestPath = f"{work_path}/videos/cctvFrames/test/"

# 画像から抽出したトレーニング特徴量の格納パス
imgFeaturesPath = f"{work_path}/videos/cctv_features/train/"

# 画像から抽出したテスト特徴量の格納パス
imgFeaturesTestPath = f"{work_path}/videos/cctv_features/test/"


# 設定値の表示
print("Training Videos (srcVideoPath): %s" % srcVideoPath)
print("Test Videos (srcTestVideoPath): %s" % srcTestVideoPath)
print("Training Images (targetImgPath): %s" % targetImgPath)
print("Test Images (targetImgTestPath): %s" % targetImgTestPath)
print("Training Images Features (imgFeaturesPath): %s" % imgFeaturesPath)
print("Test Images Features (imgFeaturesTestPath): %s" % imgFeaturesTestPath)
print("Labeled Data (labeledDataPath): %s" % labeledDataPath)
print("Training MP4 Videos (srcVideoMP4Path): %s" % srcVideoMP4Path)
print("Test MP4 Videos (srcTestVideoMP4Path): %s" % srcTestVideoMP4Path)

# COMMAND ----------

# DBTITLE 1,画像、動画表示ヘルパー関数
# displayVid(): クラウドストレージ上の動画を表示
def displayVid(filepath):
  return displayHTML("""
  <video width="480" height="320" controls>
  <source src="/files/%s" type="video/mp4">
  </video>
  """ % filepath)

# displayDbfsVid(): DBFS上の動画を表示
def displayDbfsVid(filepath):
  return displayHTML("""
  <video width="480" height="320" controls>
  <source src="/dbfs/%s" type="video/mp4">
  </video>
  """ % filepath)

# displayImg(): DBFS/クラウドストレージ上の動画を表示
def displayImg(filepath):
  dbutils.fs.cp(filepath, "FileStore/%s" % filepath)
  return displayHTML("""
  <img src="/files/%s">
  """ % filepath)
