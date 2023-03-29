import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, GPT2Model
from transformers import AdamW
from transformers.optimization import get_scheduler
from tqdm import tqdm


#简单数据集
class Dataset(torch.utils.data.Dataset):

    def __init__(self, params):
        split_size = params['max_length'] + 1
        text_path = params['text_path']

        with open(text_path, 'r') as f:
            text = f.readlines()

        self.lines = self.clean_text(text, split_size=split_size)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, i):
        return self.lines[i]
    
    @staticmethod
    def clean_text(text_straight, split_size):
        text_straight = [i.strip() for i in text_straight]
        text_straight = [i.replace('\n', '') for i in text_straight]
        text_straight = [i for i in text_straight if len(i) > 10]

        materials = []
        for line in text_straight:
            lsize = len(line)
            materials.extend([line[i:min(i+split_size, lsize)] for i in range(0, lsize, split_size)])
            if len(materials[-1]) <10:
                materials = materials[:-1]

        return materials


class CustomGenerator:
    def __init__(self, params):

        self.params = params

        #加载编码器
        self.tokenizer = AutoTokenizer.from_pretrained(self.params['model_hug_path'])

        #加载模型
        self.model = AutoModelForCausalLM.from_pretrained(self.params['model_hug_path'])

    def produce_loader(self, dataset):
        def collate_fn(data):
            data = self.tokenizer.batch_encode_plus(data,
                                            padding=True,
                                            truncation=True,
                                            max_length=self.params['max_length'],
                                            return_tensors='pt')

            data['labels'] = data['input_ids'].clone()

            return data


        #数据加载器
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.params['batch_size'],
            collate_fn=collate_fn,
            shuffle=True,
            drop_last=True,
        )

        return loader

    
    
    def generate(self, text):

        self.model = torch.load(self.params['save_model_path'])
        times = self.params['generate_times']

        #重复 times 遍
        data = self.tokenizer.batch_encode_plus([text] * times, return_tensors='pt')
        data['input_ids'] = data['input_ids'][:, :-1]
        data['attention_mask'] = torch.ones_like(data['input_ids'])
        data['token_type_ids'] = torch.zeros_like(data['input_ids'])
        data['labels'] = data['input_ids'].clone()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(device)

        def generate_loop(data):
            with torch.no_grad():
                for k in data.keys():
                    data[k] = data[k].to(device)
                out = self.model(**data)

            #取最后一个字
            #[5, b, 50257]
            out = out['logits']
            #[5, 50257]
            out = out[:, -1]

            #第50大的值,以此为分界线,小于该值的全部赋值为负无穷
            #[5, 50257] -> [5, 50]
            topk_value = torch.topk(out, 50).values
            #[5, 50] -> [5] -> [5, 1]
            topk_value = topk_value[:, -1].unsqueeze(dim=1)

            #赋值
            #[5, 50257]
            out = out.masked_fill(out < topk_value, -float('inf'))

            #不允许写特殊符号
            out[:, self.tokenizer.sep_token_id] = -float('inf')
            out[:, self.tokenizer.unk_token_id] = -float('inf')
            out[:, self.tokenizer.pad_token_id] = -float('inf')
            for i in '，。':
                out[:, self.tokenizer.get_vocab()[i]] = -float('inf')

            #根据概率采样,无放回,所以不可能重复
            #[5, 50257] -> [5, 1]
            out = out.softmax(dim=1)
            out = out.multinomial(num_samples=1)

            data['input_ids'] = torch.cat([data['input_ids'], out], dim=1)
            data['attention_mask'] = torch.ones_like(data['input_ids'])
            data['token_type_ids'] = torch.zeros_like(data['input_ids'])
            data['labels'] = data['input_ids'].clone()

            if data['input_ids'].shape[1] >= self.params['generate_size']:
                return data

            return generate_loop(data)

        data = generate_loop(data)

        for i in range(times):
            print(i, self.tokenizer.decode(data['input_ids'][i]))

        return 


    #训练
    def train(self, dataset):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(device)

        loader = self.produce_loader(dataset)

        optimizer = AdamW(self.model.parameters(), lr=5e-5)
        scheduler = get_scheduler(name='linear',
                                num_warmup_steps=0,
                                num_training_steps=len(loader),
                                optimizer=optimizer)

        self.model.train()
        for i, data in tqdm(enumerate(loader)):
            for k in data.keys():
                data[k] = data[k].to(device)
            out = self.model(**data)
            loss = out['loss']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            optimizer.zero_grad()
            self.model.zero_grad()

            if i % 1000 == 0:
                labels = data['labels'][:, 1:]
                out = out['logits'].argmax(dim=2)[:, :-1]

                select = labels != 0
                labels = labels[select]
                out = out[select]
                del select

                accuracy = (labels == out).sum().item() / labels.numel()

                lr = optimizer.state_dict()['param_groups'][0]['lr']

                print(i, loss.item(), lr, accuracy)

        self.model = self.model.to('cpu')
        torch.save(self.model, self.params['save_model_path'])
    
    def load_model(self, model_path):
        self.model = torch.load(model_path)


if __name__ == '__main__':
    text_path = 'xueqiu_1000.txt'
    cg = CustomGenerator()

    dataset = Dataset(text_path)
    cg.train(dataset[:100])

    cg.load_model('custom.model')
    cg.generate('今天天气真好', times=5, limit=100)