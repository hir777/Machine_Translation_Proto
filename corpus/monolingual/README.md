# monolingual ディレクトリ

このディレクトリには、機械翻訳用のコーパスが保存されています。

これらのコーパスは、平行コーパスではなく、Back-Translationにおいて対応する言語の文を生成するために用いられます。

よって、enファイルとjaファイルが同時に存在していたとしても、それらの間に関係はなく、それぞれ独立に異なる情報入手元から集めれたデータということになります。

Back-Translationによって作成された平行コーパス（一方の言語のデータは、人間によって書かれた文から成る。もう一方の言語のデータは、学習済みの機械翻訳システムによって生成された文から成る。）

## 拡張子

拡張子が en のファイルには英文が保存されています。

また、ja が拡張子として設定されているファイルには和文が保存されています。


## ファイルの種類

ファイルの種類としては、Back-Translationによる対訳生成用の元となるデータ(monotext)の一種類のみになります。

一つのファイルのサイズが大きすぎる場合は、いくつかのファイルに小分けにして保存されています。

例 monotext.ja => monotext1.ja monotext2.ja monotext3.ja
