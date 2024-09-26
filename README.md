# Fake-News-Article
- Một số bài báo giả có xu hướng sử dụng các thuật ngữ tương đối thường xuyên nhằm kích động sự phẫn nộ và kỹ năng viết trong những bài báo này thường thấp hơn đáng kể so với tin tức chuẩn.
- Phát hiện các bài báo tin giả bằng cách phân tích các mẫu viết trong bài.
- Được thực hiện bằng cách tinh chỉnh mô hình BERT.
- Đạt độ chính xác 80% trên tập dữ liệu tùy chỉnh.

## Cài đặt

- Toàn bộ `mã nguồn` đều cần phải được chạy.
- Thứ tự chạy:
- Đối với Back-End
  + Chạy 2 file code `scrape_media.py` và `scrape_politifact.py` để "cào" ra 2 file dataset cần thiết là "pre_media.csv" và "politifact_text.csv" từ 2 file dữ liệu thô là "Interactive Media Bias Chart - Ad Fontes Media.csv" và "data = pd.read_csv("politifact_data.csv")
"
  + Sau khi có được 2 file dataset là "pre_media.csv" và "politifact_text.csv" -> Tiến hành chạy file code `train.py` để thực hiện việc huấn luyện mô hình.
  + Sau khi tạo được file mô hình là `nb_state256.pth`, tiến hành lưu lại mô hình.
  + Chạy file `Fact_Check.py` để khởi tạo API Google Fact Check
  + Chạy file `inference.py` để chạy chương trình cuối sau khi đã hoàn thành các bước trên

- Đối với Front-End
  + Tiến hành mở folder Front-End bằng Visual Studio Code
  + Mở terminal và nhập dòng lệnh `npm start` để khởi động chương trình
### Clone

- Để clone dự án này về máy của bạn, hãy dùng `https://github.com/duyvietworkhard/Fake-News-Article-2024.git`

### Setup

- Cài đặt các thư viện/gói cần thiết.

```shell
 pip install pandas numpy scikit-learn bs4
 pip install torch
 pip install keras
 pip install pytorch_pretrained_bert
 pip install transformers
```
## Dataset

- Dữ liệu được thu thập bằng cách "cào" các trang web của những nguồn xuất bản tin tức phổ biến..
- Các bài báo thu thập được được đánh giá dựa trên các thang điểm, chất lượng và thiên kiến, theo các chỉ số thu thập từ [Politilact](https://www.politifact.com/) và [Media Charts](https://www.adfontesmedia.com/interactive-media-bias-chart/?v=402f03a963ba).
- Một số bước tiền xử lý cơ bản cũng được thực hiện trên văn bản thu thập từ việc quét các trang web.

### Tiền xử lý
- Sử dụng [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) để quét các bài báo từ web. Beautiful Soup là một thư viện Python được thiết kế cho các dự án nhanh như quét màn hình
- Cũng sử dụng một số hàm tự tạo để loại bỏ dấu câu, v.v..
![](https://miro.medium.com/max/495/1*AaAIETIq7XNlLrFQW7BtZg.png)

> Quét từ các trang web được liệt kê trong [politifact_data.csv](https://github.com/abhilashreddys/Fake-News-Article/blob/master/politifact_data.csv)
```terminal
python scrape_politifact.py
```

> Quét từ các trang web được liệt kê trong [Interactive Media Bias Chart - Ad Fontes Media.csv](https://github.com/abhilashreddys/Fake-News-Article/blob/master/Interactive%20Media%20Bias%20Chart%20-%20Ad%20Fontes%20Media.csv)
```terminal
python scrape_media.py
```
- Dữ liệu sau khi quét và tiền xử lý sẽ ra được 2 file [politifact_text.csv] và [pre_media.csv]

## Mô hình
- Được huấn luyện bằng cách tinh chỉnh mô hình BERT
- Sử dụng [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) với tinh chỉnh
- BERT, viết tắt của Bidirectional Encoder Representations from Transformers.
- BERT được thiết kế để huấn luyện trước các biểu diễn sâu hai chiều từ văn bản chưa gán nhãn bằng cách đồng thời điều kiện hóa ngữ cảnh từ cả hai hướng trái và phải ở tất cả các tầng. Kết quả là, mô hình BERT đã được huấn luyện trước có thể được tinh chỉnh với chỉ một lớp đầu ra bổ sung để tạo ra các mô hình tiên tiến nhất cho nhiều tác vụ, chẳng hạn như trả lời câu hỏi và suy luận ngôn ngữ, mà không cần phải thay đổi cấu trúc chuyên biệt cho từng tác vụ.
![](https://github.com/manideep2510/siamese-BERT-fake-news-detection-LIAR/blob/master/doc_images/bert.png?raw=true)

```python
class BertBinaryClassifier(nn.Module):
    def __init__(self, dropout=0.1):
        super(BertBinaryClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, tokens, masks=None):
        _, pooled_output = self.bert(tokens, attention_mask=masks, output_all_encoded_layers=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        proba = self.sigmoid(linear_output)
        return proba
```



## Chạy kết quả

- Run `inference.py` and mention url of the article you want to test in comand line


## Lưu ý và gợi ý

- Kiểm tra đúng vị trí các tệp, thay đổi nếu cần thiết.
- File python3 (.ipynb) chỉ để dùng thử khi không chạy được file Python [transfrom_spam.ipynb] dùng để huấn luyện và [fake_article.ipynb] để chạy kết quả.
- Chỉ được huấn luyện trong `5 Epochs`, hãy thử sử dụng một mô hình tốt hơn với nhiều dữ liệu hơn..

## Tài liệu tham khảo

- Với dữ liệu [Politilact](https://www.politifact.com/) và [Media Charts](https://www.adfontesmedia.com/interactive-media-bias-chart/?v=402f03a963ba)
- [Keras: The Python Deep Learning library](https://keras.io)
- [A library of state-of-the-art pretrained models for Natural Language Processing](https://github.com/huggingface/pytorch-transformers)
- [Pytorch Deep Learning framework](https://github.com/pytorch/pytorch)
- [Pytorch BERT usage example](https://github.com/sugi-chan/custom_bert_pipeline)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language
Understanding](https://arxiv.org/abs/1810.04805)

```bibtex
@article{Wolf2019HuggingFacesTS,
  title={HuggingFace's Transformers: State-of-the-art Natural Language Processing},
  author={Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and R'emi Louf and Morgan Funtowicz and Jamie Brew},
  journal={ArXiv},
  year={2019},
  volume={abs/1910.03771}
}
```
```bibtex
@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
```
## Trang web có thể bạn sẽ muốn ghé qua (Nếu muốn phát triển đề tài theo hướng tiếng Việt)

- https://vietfactcheck.org/
- https://tingia.gov.vn/

## Tham khảo các đề tài tương tự.

- [Triple Branch BERT Siamese Network for fake news classification on LIAR-PLUS dataset](https://github.com/manideep2510/siamese-BERT-fake-news-detection-LIAR)
- [Fake News Detection by Learning Convolution Filters through Contextualized Attention](https://github.com/ekagra-ranjan/fake-news-detection-LIAR-pytorch)
- [Based on Click-Baits](https://github.com/addy369/Click-Bait-Model)
- [Fake News Web](https://github.com/addy369/Fake_News_Web)
- [Fake News Pipeline Project](https://github.com/walesdata/newsbot), Explained article [here](https://towardsdatascience.com/full-pipeline-project-python-ai-for-detecting-fake-news-with-nlp-bbb1eec4936d).

# Một số đường dẫn bài báo test

- https://www.statesman.com/story/news/politics/politifact/2024/08/04/politifact-claim-that-trump-would-cut-social-security-lacks-basis/74647925007/
- https://www.aol.com/kamala-harris-repeats-dubious-claim-100434778.html
- https://www.statesman.com/story/news/politics/politifact/2024/09/14/trump-repeats-baseless-claims-that-haitian-immigrants-are-eating-pets/75211191007/
- https://www.statesman.com/story/news/factcheck/2024/09/20/haitian-immigrants-trump-fact-check/75304414007/
- https://www.statesman.com/story/news/politics/politifact/2024/09/02/how-much-would-donald-trumps-proposed-tariffs-cost-typical-families/75043631007/
- https://www.statesman.com/story/news/politics/elections/2024/09/20/voting-us-presidential-election-begins/75289031007/