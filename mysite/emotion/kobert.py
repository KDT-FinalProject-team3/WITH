from datetime import datetime

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
import gluonnlp as nlp
import numpy as np

# kobert
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model


#전처리 클래스
from .models import EmotionResult


class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i],))

    def __len__(self):
        return (len(self.labels))


# 감정분류 클래스
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=11,  # 클래스 수 조정
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)


class KoBERT:
    def __init__(self, txt_file):
        self.txt_file = txt_file
        # BERT 모델, Vocabulary 불러오기
        self.device = torch.device('cpu')
        self.tokenizer = get_tokenizer()
        self.bertmodel, self.vocab = get_pytorch_kobert_model()
        self.model = BERTClassifier(self.bertmodel, dr_rate=0.5).to(self.device)
        self.model.load_state_dict(
            torch.load('C:/finalproject/team3_project/mysite/emotion/model/model_state_dict.pt', map_location=self.device))
        self.model.eval()

        # 토큰화
        self.tok = nlp.data.BERTSPTokenizer(self.tokenizer, self.vocab, lower=False)

    def result_figure(self, file_name):
        print(file_name[-3:])
        if file_name[-3:] == 'txt':
            print("txt")
            predict_sentence = self.txt_file.read().decode('utf-8')
            result_figure = self.predict(predict_sentence)
            self.object_figure(result_figure, datetime.now())

            return result_figure

        elif file_name[-3:] == 'csv':
            result_list = []
            fine_data = pd.read_csv(self.txt_file, encoding='UTF-8')
            print("csv")
            for sentence, date in zip(fine_data['본문'], fine_data['작성일']):
                result_figure = self.predict(sentence)
                self.object_figure(result_figure, date)
                result_list.append(result_figure)

            return result_list

    def object_figure(self, result_figure, date):
        for i in range(11):
            if i == 0:
                fear = result_figure[i]
            elif i == 1:
                surprise = result_figure[i]
            elif i == 2:
                anger = result_figure[i]
            elif i == 3:
                sadness = result_figure[i]
            elif i == 4:
                neutrality = result_figure[i]
            elif i == 5:
                happiness = result_figure[i]
            elif i == 6:
                anxiety = result_figure[i]
            elif i == 7:
                embarrassed = result_figure[i]
            elif i == 8:
                hurt = result_figure[i]
            elif i == 9:
                interest = result_figure[i]
            elif i == 10:
                boredom = result_figure[i]

        EmotionResult.objects.create(user_id=1,
                                     fear=fear,
                                     surprise=surprise,
                                     anger=anger,
                                     sadness=sadness,
                                     neutrality=neutrality,
                                     happiness=happiness,
                                     anxiety=anxiety,
                                     embarrassed=embarrassed,
                                     hurt=hurt,
                                     interest=interest,
                                     boredom=boredom,
                                     date=date)

    def predict(self, predict_sentence):
        max_len = 64
        batch_size = 64

        data = [predict_sentence, '0']
        dataset_another = [data]
        another_test = BERTDataset(dataset_another, 0, 1, self.tok, max_len, True, False)
        test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)

        self.model.eval()

        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
            token_ids = token_ids.long().to(self.device)
            segment_ids = segment_ids.long().to(self.device)

            valid_length = valid_length
            label = label.long().to(self.device)

            out = self.model(token_ids, valid_length, segment_ids)

            test_eval = []
            for i in out:
                logits = i
                logits = logits.detach().cpu().numpy()

                # if np.argmax(logits) == 0:
                #     test_eval.append("공포")
                # elif np.argmax(logits) == 1:
                #     test_eval.append("놀람")
                # elif np.argmax(logits) == 2:
                #     test_eval.append("분노")
                # elif np.argmax(logits) == 3:
                #     test_eval.append("슬픔")
                # elif np.argmax(logits) == 4:
                #     test_eval.append("중립")
                # elif np.argmax(logits) == 5:
                #     test_eval.append("기쁨")
                # elif np.argmax(logits) == 6:
                #     test_eval.append("불안")
                # elif np.argmax(logits) == 7:
                #     test_eval.append("당황")
                # elif np.argmax(logits) == 8:
                #     test_eval.append("상처")
                # elif np.argmax(logits) == 9:
                #     test_eval.append("흥미")
                # elif np.argmax(logits) == 10:
                #     test_eval.append("지루함")

            # emotion_list = ["fear", "surprise", "anger", "sadness", "neutrality", "happiness", "anxiety", "embarrassed", "hurt", "interest", "boredom"]
            # emotion_dict = {i: j for i, j in zip(emotion_list, logits)}
            logits_list = list(logits)

            return logits_list
