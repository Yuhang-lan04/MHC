from __future__ import print_function, division  
import os  
import argparse  
import torch
from transformers import BertTokenizer, BertModel  
import numpy as np  
import pickle
import random  

def set_seed(seed=2000):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def expand_label_with_prompt1(label, tokenizer, bert_model):
    embeddings = []
    prompts = [
        "a clear and detailed image of a {}",
        "a well-composed photograph of a {}",
        "a high-quality picture of a {}",
        "a visually appealing shot of a {}",
        "a crisp and vibrant photo of a {}",
        "a beautifully captured image of a {}",
        "a stunning and sharp picture of a {}",
        "a professional-grade photo of a {}",
        "a striking and memorable image of a {}",
        "a captivating and clear photograph of a {}"
    ]
    for lbl in label:
        description = random.choice(prompts).format(lbl)
        embedding = encode_description([description], tokenizer, bert_model)
        print(f"Processing label: {lbl}, Description: {description}, Processed count: {len(embeddings) + 1}")
        embeddings.append(embedding)
    embeddings = np.stack(embeddings)
    return embeddings

def encode_description(description, tokenizer, bert_model):
    inputs_text = tokenizer(description, return_tensors="pt", padding="max_length", max_length=128, truncation=True).to("cuda")

    with torch.no_grad():
        outputs = bert_model(**inputs_text)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()  # 获取平均池化后的嵌入
    
    return embedding

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description='train',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', type=str, default='LandUse21')
    parser.add_argument('--dataPath', type=str, default='C:/Users/dell/Desktop/multi _view_ classification/re/MHC/data')
    parser.add_argument('--gpu', default='0', type=str, help='GPU device idx to use.')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))

    
    # 加载数据
    with open(args.dataPath + '/' + args.dataset + '/Y.pkl', 'rb') as f:
        label = pickle.load(f)
    first_nonzero_indices = [np.argmax(row != 0) for row in label]
    label = np.array(first_nonzero_indices)
    set_seed(2000)

    expanded_labels_file = args.dataPath + '/'+ args.dataset + '_expanded_labels_'+'.npy'
    if (os.path.exists(expanded_labels_file)):
            print("Expanded labels file already exists. Skipping expansion.")
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased').to("cuda")
        expanded_labels = expand_label_with_prompt1(label, tokenizer, bert_model)
        np.save(expanded_labels_file, expanded_labels)

