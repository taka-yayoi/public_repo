# Databricks notebook source
# MAGIC %md
# MAGIC # HTML、Javascript、D3、SVGサンプルノートブック
# MAGIC 
# MAGIC Javascript、D3のようなHTMLコードやSVGを参照するには`displayHTML`メソッドを使用します。
# MAGIC 
# MAGIC **注意:** 
# MAGIC 
# MAGIC - コンテンツと出力を含むノートブックセルの最大サイズは16MBです。`displayHTML`関数に引き渡すHTMLのサイズがこの値を超えないようにしてください。
# MAGIC - 外部リソースにリンクする場合には`http://`ではなく`https://`を使用してください。さもないと、混成コンテンツエラーのため、画像、グラフィック、Javascriptが適切にレンダリングされない場合があります。

# COMMAND ----------

# MAGIC %md ## HTMLコードの表示

# COMMAND ----------

displayHTML("<h3>ノートブックでHTMLコードを参照できます。</h3>")

# COMMAND ----------

# MAGIC %md ## SVGビジュアルの表示

# COMMAND ----------

displayHTML("""<svg width="100" height="100">
   <circle cx="50" cy="50" r="40" stroke="green" stroke-width="4" fill="yellow" />
   お使いのブラウザはインラインSVGをサポートしていません。
</svg>""")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## displayHTMLによるD3ビジュアライゼーションの表示
# MAGIC 
# MAGIC **注意:** D3ビジュアライゼーションのHTMLコードの一部をRDDからプログラムすることが可能です。
# MAGIC 
# MAGIC D3の詳細に関しては http://d3js.org/ を参照ください。

# COMMAND ----------

# D3ビジュアライゼーションを変更するためにお好きな色に変更してください
colorsRDD = sc.parallelize([(197,27,125), (222,119,174), (241,182,218), (253,244,239), (247,247,247), (230,245,208), (184,225,134), (127,188,65), (77,146,33)])
colors = colorsRDD.collect()

# COMMAND ----------

htmlCode = """
<!DOCTYPE html>
<meta charset="utf-8">
<style>

path {{
  fill: yellow;
  stroke: #000;
}}

circle {{
  fill: #fff;
  stroke: #000;
  pointer-events: none;
}}

.PiYG .q0-9{{fill:rgb{colorArray[0]}}}
.PiYG .q1-9{{fill:rgb{colorArray[1]}}}
.PiYG .q2-9{{fill:rgb{colorArray[2]}}}
.PiYG .q3-9{{fill:rgb{colorArray[3]}}}
.PiYG .q4-9{{fill:rgb{colorArray[4]}}}
.PiYG .q5-9{{fill:rgb{colorArray[5]}}}
.PiYG .q6-9{{fill:rgb{colorArray[6]}}}
.PiYG .q7-9{{fill:rgb{colorArray[7]}}}
.PiYG .q8-9{{fill:rgb{colorArray[7]}}}

</style>
<body>
<script src="https://d3js.org/d3.v3.min.js"></script>
<script>

width = 960, height = 500;

vertices = d3.range(100).map(function(d) {{
  return [Math.random() * width, Math.random() * height];
}});

svg = d3.select("body").append("svg")
    .attr("width", width)
    .attr("height", height)
    .attr("class", "PiYG")
    .on("mousemove", function() {{ vertices[0] = d3.mouse(this); redraw(); }});

path = svg.append("g").selectAll("path");

svg.selectAll("circle")
    .data(vertices.slice(1))
  .enter().append("circle")
    .attr("transform", function(d) {{ return "translate(" + d + ")"; }})
    .attr("r", 2);

redraw();

function redraw() {{
  path = path.data(d3.geom.delaunay(vertices).map(function(d) {{ return "M" + d.join("L") + "Z"; }}), String);
  path.exit().remove();
  path.enter().append("path").attr("class", function(d, i) {{ return "q" + (i % 9) + "-9"; }}).attr("d", String);
}}

</script>
 """.format(colorArray = colors)
displayHTML (htmlCode)

# COMMAND ----------

# MAGIC %md
# MAGIC # END

# COMMAND ----------


