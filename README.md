Notion link: https://app.notion.com/p/BIR_project2-3f0b0df373c08163bb6af79457146ff6?source=copy_link

BIR Project 2 — 規則與演算法簡介

1.大量匯入：到 Manage Articles，先選 Source（PubMed 或 arXiv），
再填主題名稱、搜尋條件與篇數（10–1,000，預設 500），不需自行尋找 ID。
主題名稱會產生對應來源的 title/abstract 查詢；手動改過後不會被自動覆蓋。
arXiv 可使用分類查詢，例如 cat:cs.AI。

2.指定文章：按 Add Articles，先選 Source，再貼上多筆 PMID 或 arXiv ID。
一次最多 200 個 ID，支援空白、逗號、分號或換行。
arXiv 支援 2010.11929、1706.03762、hep-th/9901001，以及 arxiv.org 的
/abs/ 和 /pdf/ 連結。版本號會移除，同一篇的不同版本不會重複加入。
已有或先前刪除的文章不會重新加入。

3.arXiv 僅匯入 title、abstract 與識別資料，不下載 PDF 或全文。
內部 ID 使用 ARXIV:2010.11929；PubMed 仍使用 PMID 前綴，避免兩者衝突。
搜尋、Zipf A–D、Compare Domains、Word2Vec、PCA 與 CSV 匯出沿用原流程。
arXiv API 呼叫有間隔與有限次重試；若服務暫時無法使用，會顯示失敗原因，
不會為此清除既有文章。

Project #2 新增功能：

1. 網站上方新增「Zipf／詞彙分布」頁面。
2. Zipf 頁只使用 abstract，標題與正文不計入 Project #2 分析。
3. 比較四種累積 preprocessing：
   A：空白斷詞＋小寫
   B：移除標點並使用專案 tokenizer
   C：移除 stopwords
   D：Porter stemming
4. 顯示 documents、total tokens、unique terms、average tokens/document。
5. 顯示 Rank vs Frequency 與 Log-Log Plot。
6. 線性迴歸會報告 slope、intercept、Zipf exponent、R²、RMSE。
7. 顯示 Top 50 terms 的 CF、DF、IDF，並可下載完整 CSV。
8. 「單一詞分布搜尋」可查一個詞在每篇摘要中的 TF，並顯示整體 CF、DF、IDF。結果會在獨立頁面顯示，可在頁面頂端直接改查其他單詞。
9. 一般文獻搜尋若只輸入一個詞，結果頁也會直接顯示摘要 collection 的 CF、DF、IDF，並連到完整分布。
10. 除 PMC JATS 全文 XML 外，現在也可上傳 PubMed EFetch XML；一個 XML 可以包含多篇 PubmedArticle records。
11. 也可直接上傳先前 Project #2 下載程式產生的 data/pubmed.json，不必重新下載或轉檔。
12. 「PMID查詢/上傳」可輸入 PubMed 主題並批次加入 10～1,000 篇摘要；欄位只顯示
    GLP-1[Title/Abstract] 作為提示範例，不會自動填入，結果依 PubMed relevance 排序，重複 PMID 會自動略過。
13. 支援多個命名主題；每個主題的 Zipf、CF、DF、IDF 與搜尋索引完全分開。
14. Word2Vec 可選 CBOW 或 Skip-gram，兩者都使用 Negative Sampling，並可查相似詞。
15. 可調整 window size（1～5）、vector dimension（8～64）與 epochs（1～5）；預設為 2、24、2。
16. 搜尋加入 Dynamic Programming Levenshtein Edit Distance 拼字修正，會顯示「你是不是要找……」。
17. 已有文章清單可以逐篇刪除；刪除後會立即重建編號、搜尋索引、Zipf 與 Word2Vec 分析。
18. Word2Vec 訓練結果新增互動式 PCA 詞向量分布圖；滑鼠移到資料點可查看詞、PC1、PC2、corpus frequency 與 cosine similarity。PCA 使用純 Python 計算，不需額外安裝套件。
19. Zipf 的線性與 log-log 圖新增滑鼠資料點提示，可查看 word、rank、frequency 與 log 座標；頁面移除重複的 collection 統計卡及非必要的高／中／低頻區段表格。
