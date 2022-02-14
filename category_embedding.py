from nltk.tokenize import word_tokenize
import re
import torch.nn as nn
import nltk
nltk.download('punkt')

class Category_embedding:
    def __init__(self,meta_path):
        self.data = pd.read_csv(meta_path)
    
    def delete_spec_chars(input): #function to delete special characters
        regex = r'[^a-zA-Z0-9\s]'
        output = re.sub(regex,'',input)    
        return output

    def preprocess(category):
        cat = delete_spec_chars(category)  
        tokens = word_tokenize(cat)  
        tokens = [token.lower() for token in tokens]
        return tokens

    def l2_norm(input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        print(buffer.size())
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def get_embeddings(self):
        data = self.data
        cat = data['category'].unique()
        cat_processed = [category.split('|')[-1] for category in cat]
        vocab = [] 
        for cat in cat_processed:
            words = preprocess(cat)  
            vocab.extend(words)
        dict_index = {}
        for i in range(len(vocab)):
            dict_index[vocab[i]] = i 
        embeds = nn.Embedding(len(vocab), 128)  
        lookup_tensor = torch.tensor(list(dict_index.values()), dtype=torch.long)
        hello_embed = embeds(lookup_tensor)
        embeddings = {} 
        count = 0
        for cat in cat_processed:
            cat_string = delete_spec_chars(cat)
            cat_string = ' '.join(cat_string.split())
            word_length = len(cat_string.split(' '))
            # print(type(hello_embed[count]))  
            if word_length == 1:
                embeddings[cat] = hello_embed[count].detach().numpy()
            else:
                emb = []
                for i in range(word_length):
                    emb.append(hello_embed[count+i].detach().numpy())
                emb = np.array(emb)
                emb = np.average(emb, axis=0)
                embeddings[cat] = emb
            count += word_length
        return embeddings



                
