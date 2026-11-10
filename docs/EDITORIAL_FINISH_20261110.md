# 写作与图形精修

本轮以已确认的实验结果为基础，精修全文英文与全部 26 张图，所有工作均在本地完成。

正文先陈述发现，再解释机制、给出比较。摘要直接收束到实际部署结果；引言围绕
读出失效、更新归因和兼容性保留展开。附录压缩重复的协议说明和防御性结尾，必要
假设、未通过的门槛、精度区别与数据范围仍保留在相应位置。正文没有用加粗句子
充当段落标题。Related Work 补充 AlphaEdit 的零空间模型编辑工作。

图形采用统一的 Arial 字体、数学字形与蓝／青／玫红配色，细化轴线、面板编号和
图例层次。机制图使用带厚度的叠层截面；主图展开 ResNet 残差块和 ViT 两个注意力
块。定量图使用平面坐标，保留全部原始种子、误差区间、坐标与统计口径。PDF/SVG
中的文字仍可编辑。

对照前一版本的完整性报告见
[EDITORIAL_VISUAL_VERIFICATION_20261110.json](EDITORIAL_VISUAL_VERIFICATION_20261110.json)。
39 个编号公式、28 个数学陈述、65 个表格主体和所有数值宏均一致；25 张图的数值
绘图对象指纹完全一致。剩下一张主图增加了局部结构轴，其定量绘图函数保持一致。
实验结果文件未改动。解压后的源码检查、图形重建与 PDF 对照结果见
[REVIEW_CHECKS.json](../REVIEW_CHECKS.json)。

参考的是论证与图形表达原则，保留当前会议格式：

- [Learning Dynamics of LLM Finetuning](https://arxiv.org/html/2407.10490)：从具体观察进入机制与验证。
- [AlphaEdit](https://proceedings.iclr.cc/paper_files/paper/2025/hash/29c8c615b3187ee995029284702d3f43-Abstract-Conference.html)：直接呈现问题、构造与证据。
- [Nature artwork guidance](https://www.nature.com/documents/nature-final-artwork.pdf)：统一字体、最终尺寸可读性和可编辑矢量输出。

用户提供的两张参考图用于借鉴叠层、局部展开和信息流组织。
