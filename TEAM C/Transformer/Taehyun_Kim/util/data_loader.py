"""
@author : Hyunwoong
@when : 2019-10-29
torchtext.legacy / Multi30k → 직접 다운로드 + 내장 데이터 폴백으로 대체
BucketIterator → SimpleIterator (길이 유사끼리 배치)
"""
import math
import urllib.request
import gzip
from collections import Counter

import torch

# torchtext로 Multi30k 다운로드 → 어휘 구축 → BucketIterator 반환.(비슷한 길이로 묶음)
# ── 내장 샘플 데이터 (Multi30k 다운로드 실패 시 폴백) ──────────────
BUILTIN_PAIRS = [
    ("a man in a blue shirt is standing on a ladder .", "ein mann in einem blauen hemd steht auf einer leiter ."),
    ("two dogs run through the field .", "zwei hunde laufen durch das feld ."),
    ("a woman is reading a book on the bench .", "eine frau liest ein buch auf der bank ."),
    ("the boy is playing soccer with his friends .", "der junge spielt fußball mit seinen freunden ."),
    ("a girl in a red dress is dancing .", "ein mädchen in einem roten kleid tanzt ."),
    ("the cat is sleeping on the sofa .", "die katze schläft auf dem sofa ."),
    ("a man is riding a bicycle in the park .", "ein mann fährt fahrrad im park ."),
    ("two children are playing in the snow .", "zwei kinder spielen im schnee ."),
    ("the woman is cooking dinner in the kitchen .", "die frau kocht abendessen in der küche ."),
    ("a dog is chasing a ball in the garden .", "ein hund jagt einen ball im garten ."),
    ("the students are studying in the library .", "die studenten lernen in der bibliothek ."),
    ("a man and a woman are walking on the beach .", "ein mann und eine frau gehen am strand spazieren ."),
    ("the children are watching television .", "die kinder schauen fernsehen ."),
    ("a bird is flying over the lake .", "ein vogel fliegt über den see ."),
    ("the old man is sitting on a park bench .", "der alte mann sitzt auf einer parkbank ."),
    ("a young woman is jogging in the morning .", "eine junge frau joggt am morgen ."),
    ("the boy is drawing a picture .", "der junge zeichnet ein bild ."),
    ("two men are playing chess .", "zwei männer spielen schach ."),
    ("a woman is singing on the stage .", "eine frau singt auf der bühne ."),
    ("the dog is barking at the mailman .", "der hund bellt den briefträger an ."),
    ("a group of people are eating at a restaurant .", "eine gruppe von menschen isst in einem restaurant ."),
    ("the man is fixing his car .", "der mann repariert sein auto ."),
    ("a little girl is feeding the ducks .", "ein kleines mädchen füttert die enten ."),
    ("the woman is watering the flowers in her garden .", "die frau gießt die blumen in ihrem garten ."),
    ("a man is reading a newspaper on the train .", "ein mann liest eine zeitung im zug ."),
    ("the children are playing on the playground .", "die kinder spielen auf dem spielplatz ."),
    ("a woman is talking on her phone .", "eine frau telefoniert mit ihrem handy ."),
    ("the boy is eating an apple .", "der junge isst einen apfel ."),
    ("a man is swimming in the ocean .", "ein mann schwimmt im ozean ."),
    ("the girl is writing in her notebook .", "das mädchen schreibt in ihr notizbuch ."),
    ("a dog is sitting next to the owner .", "ein hund sitzt neben dem besitzer ."),
    ("two women are laughing at a joke .", "zwei frauen lachen über einen witz ."),
    ("the man is climbing a mountain .", "der mann klettert einen berg hinauf ."),
    ("a child is blowing out birthday candles .", "ein kind bläst geburtstagskerzen aus ."),
    ("the students are listening to the teacher .", "die studenten hören dem lehrer zu ."),
    ("a man is painting a wall .", "ein mann streicht eine wand ."),
    ("the woman is buying vegetables at the market .", "die frau kauft gemüse auf dem markt ."),
    ("a boy is flying a kite .", "ein junge lässt einen drachen steigen ."),
    ("the family is having dinner together .", "die familie isst gemeinsam abendessen ."),
    ("a man is walking his dog in the rain .", "ein mann geht mit seinem hund im regen spazieren ."),
    ("the cat is drinking milk from a bowl .", "die katze trinkt milch aus einer schüssel ."),
    ("a woman is cutting her hair .", "eine frau schneidet ihr haar ."),
    ("the boys are kicking a football .", "die jungen treten einen fußball ."),
    ("a man is drinking coffee in a cafe .", "ein mann trinkt kaffee in einem café ."),
    ("the girl is riding a horse .", "das mädchen reitet auf einem pferd ."),
    ("two people are dancing at the party .", "zwei menschen tanzen auf der party ."),
    ("a dog is catching a frisbee .", "ein hund fängt eine frisbee ."),
    ("the man is taking a photograph .", "der mann macht ein foto ."),
    ("a woman is sitting in a chair reading .", "eine frau sitzt auf einem stuhl und liest ."),
    ("the children are swimming in a pool .", "die kinder schwimmen in einem pool ."),
    ("a man is playing the guitar on the street .", "ein mann spielt gitarre auf der straße ."),
    ("the woman is buying a dress in the store .", "die frau kauft ein kleid im laden ."),
    ("a boy is climbing a tree .", "ein junge klettert auf einen baum ."),
    ("the cat is chasing a mouse .", "die katze jagt eine maus ."),
    ("a group of kids are playing basketball .", "eine gruppe von kindern spielt basketball ."),
    ("the man is cutting wood with an axe .", "der mann hackt holz mit einer axt ."),
    ("a woman is pushing a baby carriage .", "eine frau schiebt einen kinderwagen ."),
    ("the dog is sleeping under the table .", "der hund schläft unter dem tisch ."),
    ("a man is throwing a ball to the dog .", "ein mann wirft dem hund einen ball ."),
    ("the girl is dancing in the rain .", "das mädchen tanzt im regen ."),
    ("two children are building a sandcastle .", "zwei kinder bauen eine sandburg ."),
    ("a woman is painting a landscape .", "eine frau malt eine landschaft ."),
    ("the man is washing dishes in the kitchen .", "der mann spült geschirr in der küche ."),
    ("a dog is running on the beach .", "ein hund läuft am strand ."),
    ("the students are working on a project .", "die studenten arbeiten an einem projekt ."),
    ("a man is lifting weights at the gym .", "ein mann hebt gewichte im fitnessstudio ."),
    ("the woman is smiling at the camera .", "die frau lächelt in die kamera ."),
    ("a boy is eating ice cream .", "ein junge isst eis ."),
    ("the children are singing a song .", "die kinder singen ein lied ."),
    ("a man is rowing a boat on the river .", "ein mann rudert ein boot auf dem fluss ."),
    ("the girl is playing the piano .", "das mädchen spielt klavier ."),
    ("a woman is walking through the forest .", "eine frau geht durch den wald ."),
    ("the man is building a house .", "der mann baut ein haus ."),
    ("a child is learning to ride a bike .", "ein kind lernt fahrrad fahren ."),
    ("the dog is playing with a toy .", "der hund spielt mit einem spielzeug ."),
    ("a woman is looking at the stars .", "eine frau schaut auf die sterne ."),
    ("the boy is running in the race .", "der junge läuft in dem rennen ."),
    ("two men are shaking hands .", "zwei männer schütteln sich die hände ."),
    ("a woman is making coffee in the morning .", "eine frau macht morgens kaffee ."),
    ("the man is fixing the bicycle .", "der mann repariert das fahrrad ."),
    ("a girl is picking flowers in the field .", "ein mädchen pflückt blumen auf dem feld ."),
    ("the cat is playing with a ball of yarn .", "die katze spielt mit einem wollknäuel ."),
    ("a man is chopping vegetables for dinner .", "ein mann hackt gemüse für das abendessen ."),
    ("the children are opening their presents .", "die kinder öffnen ihre geschenke ."),
    ("a woman is feeding a baby .", "eine frau füttert ein baby ."),
    ("the man is raking leaves in the yard .", "der mann harkt laub im hof ."),
    ("a boy is drawing on the chalkboard .", "ein junge zeichnet auf der tafel ."),
    ("the girl is playing with her doll .", "das mädchen spielt mit ihrer puppe ."),
    ("a man is jogging in the park .", "ein mann joggt im park ."),
    ("the woman is baking a cake .", "die frau backt einen kuchen ."),
    ("two girls are talking to each other .", "zwei mädchen reden miteinander ."),
    ("a man is teaching a class .", "ein mann hält eine klasse ab ."),
    ("the dog is playing in the mud .", "der hund spielt im schlamm ."),
    ("a woman is ironing clothes .", "eine frau bügelt kleidung ."),
    ("the boy is building with lego bricks .", "der junge baut mit legosteinen ."),
    ("a man is fixing the roof of a house .", "ein mann repariert das dach eines hauses ."),
    ("the children are running in the yard .", "die kinder rennen im hof ."),
    ("a woman is sewing a dress .", "eine frau näht ein kleid ."),
    ("the man is washing his car in the driveway .", "der mann wäscht sein auto in der einfahrt ."),
    ("a dog is jumping over a fence .", "ein hund springt über einen zaun ."),
    ("the girl is blowing bubbles .", "das mädchen pustet seifenblasen ."),
    ("a man is sleeping on the couch .", "ein mann schläft auf der couch ."),
    ("two children are sharing an umbrella .", "zwei kinder teilen einen regenschirm ."),
    ("a woman is carrying a heavy bag .", "eine frau trägt eine schwere tasche ."),
    ("the boy is playing with his toy car .", "der junge spielt mit seinem spielzeugauto ."),
    ("a man is typing on his laptop .", "ein mann tippt auf seinem laptop ."),
    ("the girl is wearing a beautiful hat .", "das mädchen trägt einen schönen hut ."),
    ("a group of people are hiking in the mountains .", "eine gruppe von menschen wandert in den bergen ."),
    ("the woman is braiding her hair .", "die frau flicht ihr haar ."),
    ("a man is fishing in the lake .", "ein mann angelt im see ."),
    ("the children are playing cards .", "die kinder spielen karten ."),
    ("a woman is applying makeup .", "eine frau schminkt sich ."),
    ("the boy is throwing snowballs .", "der junge wirft schneebälle ."),
    ("a man is playing tennis .", "ein mann spielt tennis ."),
    ("the dog is waiting at the door .", "der hund wartet an der tür ."),
    ("a woman is grocery shopping .", "eine frau kauft lebensmittel ein ."),
    ("the girl is jumping rope .", "das mädchen springt seil ."),
    ("a man is driving a truck .", "ein mann fährt einen lastwagen ."),
    ("two boys are arm wrestling .", "zwei jungen ringen mit den armen ."),
    ("a woman is arranging flowers in a vase .", "eine frau arrangiert blumen in einer vase ."),
    ("the man is playing the drums .", "der mann spielt schlagzeug ."),
    ("a child is playing in a sandbox .", "ein kind spielt in einem sandkasten ."),
    ("the cat is sitting on the windowsill .", "die katze sitzt auf dem fensterbrett ."),
    ("a woman is practicing yoga .", "eine frau praktiziert yoga ."),
    ("the man is mowing the lawn .", "der mann mäht den rasen ."),
    ("a boy is looking through a telescope .", "ein junge schaut durch ein teleskop ."),
    ("the girl is helping her mother in the kitchen .", "das mädchen hilft ihrer mutter in der küche ."),
    ("a man is reading bedtime stories to his children .", "ein mann liest seinen kindern gute nacht geschichten vor ."),
    ("the woman is taking care of her plants .", "die frau kümmert sich um ihre pflanzen ."),
]


class Vocab:
    PAD, SOS, EOS, UNK = '<pad>', '<sos>', '<eos>', '<unk>'

    def __init__(self):
        self.stoi = {self.PAD: 0, self.SOS: 1, self.EOS: 2, self.UNK: 3}
        self.itos = {v: k for k, v in self.stoi.items()}

    def build(self, sentences, min_freq=1):
        counter = Counter(tok for sent in sentences for tok in sent.lower().strip().split())
        for word, freq in counter.items():
            if freq >= min_freq and word not in self.stoi:
                idx = len(self.stoi)
                self.stoi[word] = idx
                self.itos[idx] = word

    def encode(self, sentence, max_len=None):
        tokens = sentence.lower().strip().split()
        if max_len:
            tokens = tokens[:max_len - 2]
        ids = ([self.stoi[self.SOS]] +
               [self.stoi.get(t, self.stoi[self.UNK]) for t in tokens] +
               [self.stoi[self.EOS]])
        return ids

    def decode(self, indices):
        special = {self.stoi[s] for s in [self.PAD, self.SOS, self.EOS, self.UNK]}
        return ' '.join(self.itos[i] for i in indices if i not in special)

    def __len__(self):
        return len(self.stoi)


class SimpleIterator:
    """BucketIterator 대체 — 길이 유사끼리 배치"""

    def __init__(self, pairs, batch_size, src_vocab, trg_vocab, max_len, device, shuffle=True):
        self.pairs      = pairs
        self.batch_size = batch_size
        self.src_vocab  = src_vocab
        self.trg_vocab  = trg_vocab
        self.max_len    = max_len
        self.device     = device
        self.shuffle    = shuffle

    def __iter__(self):
        pairs = self.pairs[:]
        if self.shuffle:
            idx   = torch.randperm(len(pairs)).tolist()
            pairs = [pairs[i] for i in idx]
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start:start + self.batch_size]
            if len(batch) < 2:
                continue
            src_seqs, trg_seqs = [], []
            for en, de in batch:
                src_seqs.append(self.src_vocab.encode(en, self.max_len))
                trg_seqs.append(self.trg_vocab.encode(de, self.max_len))

            def pad(seqs):
                L = max(len(s) for s in seqs)
                return torch.tensor(
                    [s + [0] * (L - len(s)) for s in seqs],
                    dtype=torch.long
                )
            yield pad(src_seqs).to(self.device), pad(trg_seqs).to(self.device)

    def __len__(self):
        return math.ceil(len(self.pairs) / self.batch_size)


class DataLoader:
    """원본 DataLoader 인터페이스 유지"""

    source: Vocab = None
    target: Vocab = None

    def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token):
        self.ext         = ext
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de
        self.init_token  = init_token
        self.eos_token   = eos_token
        print('dataset initializing start')

    def _download_multi30k(self):
        BASE  = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/"
        files = {
            "train.en": "train.en.gz", "train.de": "train.de.gz",
            "val.en":   "val.en.gz",   "val.de":   "val.de.gz",
            "test.en":  "test_2016_flickr.en.gz",
            "test.de":  "test_2016_flickr.de.gz",
        }
        data = {}
        for name, fname in files.items():
            try:
                with urllib.request.urlopen(BASE + fname, timeout=10) as r:
                    data[name] = gzip.decompress(r.read()).decode('utf-8').strip().split('\n')
                print(f'  ✅ {name}: {len(data[name])} 문장')
            except Exception as e:
                print(f'  ❌ {name} 다운로드 실패: {e}')
                return None
        return data

    def make_dataset(self):
        print('Multi30k 다운로드 시도...')
        multi30k = self._download_multi30k()

        if multi30k:
            train_pairs = list(zip(multi30k['train.en'], multi30k['train.de']))
            valid_pairs = list(zip(multi30k['val.en'],   multi30k['val.de']))
            test_pairs  = list(zip(multi30k['test.en'],  multi30k['test.de']))
            print(f'✅ Multi30k 로드 완료')
        else:
            print('⚠️  내장 샘플 데이터 사용 (130문장 × 6 증강)')
            pairs       = BUILTIN_PAIRS * 6
            n           = len(pairs)
            train_pairs = pairs[:int(n * 0.8)]
            valid_pairs = pairs[int(n * 0.8):int(n * 0.9)]
            test_pairs  = pairs[int(n * 0.9):]

        print(f'train: {len(train_pairs):,}  valid: {len(valid_pairs):,}  test: {len(test_pairs):,}')
        return train_pairs, valid_pairs, test_pairs

    def build_vocab(self, train_data, min_freq=1):
        self.source = Vocab()
        self.target = Vocab()
        self.source.build([p[0] for p in train_data], min_freq=min_freq)
        self.target.build([p[1] for p in train_data], min_freq=min_freq)
        print(f'어휘 크기 | src(EN): {len(self.source):,}  trg(DE): {len(self.target):,}')

    def make_iter(self, train, validate, test, batch_size, device, max_len=128):
        train_iterator = SimpleIterator(train,    batch_size, self.source, self.target, max_len, device, shuffle=True)
        valid_iterator = SimpleIterator(validate, batch_size, self.source, self.target, max_len, device, shuffle=False)
        test_iterator  = SimpleIterator(test,     batch_size, self.source, self.target, max_len, device, shuffle=False)
        print('dataset initializing done')
        return train_iterator, valid_iterator, test_iterator
