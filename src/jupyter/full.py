import os

if 1:
    ## for path compatibility, if you are not running from app.py, please specify the project root path as working directory (there is no __file__ in jupyter notebook)
    _root_path_ = 'D:\\_work\\Bi-Senti-EE6483'

    if '_root_path_' in locals():
        os.chdir(_root_path_)
    assert os.path.basename(os.getcwd()) == 'Bi-Senti-EE6483'

    import configparser as cp

    # storage for multiple data processing, for comparison of different methods
    from enum import IntFlag, auto
    class F(IntFlag): # data flow

        RE = auto() # need Regex to remove special characters

        TOKEN_NLTK = auto() # need tokenization by non-deeplearning methods
        TOKEN_SPACY = auto()
        TOKEN_GENSIM = auto()

        STOPWORDS_NLTK = auto() # need stopwords removal by non-deeplearning methods
        STOPWORDS_SPACY = auto()
        STOPWORDS_GENSIM = auto()
        
        LEMMATIZE_NKTK = auto() # need lemmatization by non-deeplearning methods
        LEMMATIZE_SPACY = auto()
        LEMMATIZE_TEXTBLOB = auto()

        EMBEDDING_WORD2VEC_TRAIN = auto() # shallow neural network training
        EMBEDDING_WORD2VEC_PRETRAIN = auto()
        EMBEDDING_GLOVE_PRETRAIN = auto()
        EMBEDDING_TFIDF = auto()

        MODEL_SVM = auto() # traditional machine learning
        MODEL_ELM = auto()
        MODEL_GP = auto() # gaussian process
        MODEL_RF = auto() # random forest not supported yet
        MODEL_LINEAR = auto() # OLS/ Lasso/ Ridge may not perform well in this case, not implemented
        MODEL_RNN = auto() # deep neural network training
        MODEL_LSTM = auto()
        MODEL_GRU= auto()

        ENSEMBLE_BERT = auto() # Fine-tuning pre-trained huggingface models, does not follow the taskflow definition, will be done separately
        ENSEMBLE_DISTILBERT = auto()
        ENSEMBLE_ROBERTA = auto()

        CUSTOMIZED = auto() # for customized encoder-only model training


        preset1 = RE | TOKEN_SPACY | STOPWORDS_SPACY | LEMMATIZE_SPACY | EMBEDDING_WORD2VEC_PRETRAIN | MODEL_ELM
        preset2 = ENSEMBLE_BERT

    #****************************************************************************************************
    # USER DEFINED HERE
    taskflows:list[F] = [
        F.ENSEMBLE_BERT,
        #[F.ENSEMBLE_DISTILBERT],
        F.preset1
    ]
    #****************************************************************************************************

    class DATA_CONTAINER(list):
        def __init__(self, taskflows):
            super().__init__([ [] for _i in range(len(taskflows)) ]) # empty datacontainers with the amount of taskflows
        def append(self, data):
            raise NotImplementedError('Append porhibited. Can only change sublists')
    container_traintest = DATA_CONTAINER(taskflows) # for training and testing data, empty for now
    container_pred = DATA_CONTAINER(taskflows)

if 2:
    import pandas as pd
    from copy import deepcopy

    # Read the data
    labeled_df = pd.read_json('data/train.json')
    unlabeled_df = pd.read_json('data/test.json')

    # Output the info
    print("\nTrain DataFrame info:")
    labeled_df.info()
    print("\nTest DataFrame info:")
    unlabeled_df.info()

    labeled_df = labeled_df.head(3)
    unlabeled_df = unlabeled_df.head(2)

    for i in range(len(container_traintest)):
        container_traintest[i].append(deepcopy(labeled_df))
        container_pred[i].append(deepcopy(unlabeled_df))

    assert type(container_traintest[0][0]) == pd.DataFrame

if 3:
    # for all sentences, we first apply regular expression to remove all special characters
    import re

    def re_removal(text: str) -> str:
        text=re.sub('(<.*?>)', ' ', text)
        text=re.sub('[,\.!?:()"]', '', text)
        text=re.sub('[^a-zA-Z"]',' ',text)
        return text.lower()


    # tokenizer is a function that splits a text(very long str) into words(list of str)
    def tokenize(text: str , method: str) -> list[str] :
        if method == 'split':
            return text.split()
        elif method == 'nltk':
            import nltk
            from nltk.tokenize import word_tokenize
            nltk.download('punkt')
            return word_tokenize(text)
        elif method == 'spacy':
            import spacy
            nlp_en_model = spacy.load("en_core_web_sm")
            return [token.text for token in nlp_en_model(text)] #nlp_en_model(text) returns a generator（doc), yeilds tokens, token.text is the word, token.lemma_ is the lemma, token.pos_ is the POS
        elif method == 'gensim':
            import gensim
            return gensim.utils.simple_preprocess(text)
        elif method == 'bert':
            raise ValueError('bert based tokenizer should be implemented afterwards, since it returns a different type of data')
        else:
            raise ValueError('method not supported')
            
    def remove_stopwords(text: list[str] , method: str) -> list[str]:
        '''
        text: for nltk, only a sentence contains words, not a list of sentences
            for spacy, it accept a str
            for gensim, it accept a str
        return: for nltk and spacy, a list of filtered words
                for gensim, a str
        '''
        if method == 'nltk':
            import nltk
            from nltk.corpus import stopwords
            nltk.download('stopwords')
            nltk.download('punkt')
            stop_words = set(stopwords.words('english'))
            return [word for word in text if word not in stop_words]
        elif method == 'spacy':
            import spacy
            nlp_en_model = spacy.load("en_core_web_sm")
            stop_words = {word for word in nlp_en_model.Defaults.stop_words}
            return [word for word in text if word not in stop_words]
        elif method == 'gensim':
            from gensim.parsing.preprocessing import STOPWORDS
            stopwords = set(STOPWORDS)
            return [word for word in text if word not in stop_words]
        else:
            raise ValueError('method not supported')

    def lematize(text: list[str], method: str) -> list[str]:
        '''
        text: a list of words
        return: a list of lemmatized words
        '''
        if method == 'nltk':
            import nltk
            from nltk.stem import WordNetLemmatizer
            def pos_tagger(nltk_tag):
                from nltk.corpus import wordnet
                if nltk_tag.startswith('J'):
                    return wordnet.ADJ
                elif nltk_tag.startswith('V'):
                    return wordnet.VERB
                elif nltk_tag.startswith('N'):
                    return wordnet.NOUN
                elif nltk_tag.startswith('R'):
                    return wordnet.ADV
                else:         
                    return None
            def tagged_lemma(listofstr: list[str]) -> list[str]:
                nonlocal lemmatizer
                pos_tagged = nltk.pos_tag(listofstr)
                wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
                lemmatized_sentence = []
                for word, tag in wordnet_tagged:
                    if tag is None:
                        lemmatized_sentence.append(word)
                    else:       
                        lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
                return lemmatized_sentence
            nltk.download('wordnet')
            lemmatizer = WordNetLemmatizer()
            return tagged_lemma(text)
        elif method == 'spacy':
            import spacy
            nlp_en_model = spacy.load("en_core_web_sm")
            return [token.lemma_ for token in nlp_en_model(" ".join(text))]
        elif method == 'textblob':
            from textblob import TextBlob
            blob = TextBlob(" ".join(text))
            return [word.lemmatize() for word in blob.words]
        else:
            raise ValueError('method not supported')

    for taski in range(len(taskflows)):
        if taskflows[taski] & F.RE:
            container_traintest[taski].append(container_traintest[taski][0]['reviews'].apply(re_removal))
            container_pred[taski].append(container_pred[taski][0]['reviews'].apply(re_removal))
        
        if taskflows[taski] & F.TOKEN_NLTK: # from this step, we will append each result to the container
            container_traintest[taski].append(container_traintest[taski][-1].apply(tokenize, method='nltk'))
            container_pred[taski].append(container_pred[taski][-1].apply(tokenize, method='nltk'))
        elif taskflows[taski] & F.TOKEN_SPACY:
            container_pred[taski].append(container_pred[taski][-1].apply(tokenize, method='spacy'))
            container_traintest[taski].append(container_traintest[taski][-1].apply(tokenize, method='spacy'))
        elif taskflows[taski] & F.TOKEN_GENSIM:
            container_traintest[taski].append(container_traintest[taski][-1].apply(tokenize, method='gensim'))
            container_pred[taski].append(container_pred[taski][-1].apply(tokenize, method='gensim'))
        # after this step, last item in container is list[str] for each sentence

        if taskflows[taski] & F.STOPWORDS_NLTK:
            container_traintest[taski].append(pd.Series(container_traintest[taski][-1]).apply(remove_stopwords, method='nltk'))
            container_pred[taski].append(pd.Series(container_pred[taski][-1]).apply(remove_stopwords, method='nltk'))
        elif taskflows[taski] & F.STOPWORDS_SPACY:
            container_traintest[taski].append(pd.Series(container_traintest[taski][-1]).apply(remove_stopwords, method='spacy'))
            container_pred[taski].append(pd.Series(container_pred[taski][-1]).apply(remove_stopwords, method='spacy'))
        elif taskflows[taski] & F.STOPWORDS_GENSIM:
            container_traintest[taski].append(pd.Series(container_traintest[taski][-1]).apply(remove_stopwords, method='gensim'))
            container_pred[taski].append(pd.Series(container_pred[taski][-1]).apply(remove_stopwords, method='gensim'))

        if taskflows[taski] & F.LEMMATIZE_NKTK:
            container_traintest[taski].append(pd.Series(container_traintest[taski][-1]).apply(lematize, method='nltk'))
            container_pred[taski].append(pd.Series(container_pred[taski][-1]).apply(lematize, method='nltk'))
        elif taskflows[taski] & F.LEMMATIZE_SPACY:
            container_traintest[taski].append(pd.Series(container_traintest[taski][-1]).apply(lematize, method='spacy'))
            container_pred[taski].append(pd.Series(container_pred[taski][-1]).apply(lematize, method='spacy'))
        elif taskflows[taski] & F.LEMMATIZE_TEXTBLOB:
            container_traintest[taski].append(pd.Series(container_traintest[taski][-1]).apply(lematize, method='textblob'))
            container_pred[taski].append(pd.Series(container_pred[taski][-1]).apply(lematize, method='textblob'))
if 4 :
    from gensim.models import Word2Vec, FastText
    from gensim.models.keyedvectors import KeyedVectors
    import gensim.downloader as api
    import numpy as np
    from transformers import BertTokenizer, BertModel
    import torch
    from sklearn.feature_extraction.text import TfidfVectorizer
    from typing import TypeAlias
    T_embedding: TypeAlias = list[list[np.ndarray]] #  corpus ,sentences, words-> embedding
    train_w2v_model = None # avoid deconstruction
    word2vec_model = None
    glove_model = None

    def get_embeddings(texts: list[list[str]], method: str) -> T_embedding:
        if method == 'word2vec_trained':
            train_w2v_model = Word2Vec(sentences=texts, vector_size=100, window=5, min_count=1, workers=4)
            embeddings = [[train_w2v_model.wv[word] for word in text if word in train_w2v_model.wv] for text in texts]
        elif method == 'word2vec_fromPretrained':
            word2vec_file = 'temp/GoogleNews-vectors-negative300.bin'
            word2vec_model = KeyedVectors.load_word2vec_format(word2vec_file, binary=True)
            embeddings = [[word2vec_model[word] for word in text if word in word2vec_model] for text in texts]
        elif method == 'glove_fromPretrained':
            glove_model = api.load('glove-wiki-gigaword-100')
            embeddings = [[glove_model[word] for word in text if word in glove_model] for text in texts]
        elif method == 'fasttext':
            raise NotImplementedError()
        elif method == 'bert':
            raise UserWarning('Bert has dynamic embedding, not like word2vec, fasttext, glove')
        elif method == 'tfidf':
            # when using this, make sure that 'del tfidf_vectorizer' before a new dataflow
            if not 'tfidf_vectorizer' in locals():
                tfidf_vectorizer = TfidfVectorizer()
                X = tfidf_vectorizer.fit_transform(texts)
                embeddings = X.toarray()
            else:
                embeddings = tfidf_vectorizer.transform(texts).toarray()
        else:
            raise ValueError('method not supported')
        
        return embeddings

    for taski in range(len(taskflows)):
        if taskflows[taski] & F.EMBEDDING_WORD2VEC_TRAIN:
            method='word2vec_trained'
            container_traintest[taski].append(get_embeddings(container_traintest[taski][-1].tolist(), method=method))
            container_pred[taski].append(get_embeddings(container_pred[taski][-1].tolist(), method=method))
        elif taskflows[taski] & F.EMBEDDING_WORD2VEC_PRETRAIN:
            method='word2vec_fromPretrained'
            container_traintest[taski].append(get_embeddings(container_traintest[taski][-1].tolist(), method=method))
            container_pred[taski].append(get_embeddings(container_pred[taski][-1].tolist(), method=method))
        elif taskflows[taski] & F.EMBEDDING_GLOVE_PRETRAIN:
            method='glove_fromPretrained'
            container_traintest[taski].append(get_embeddings(container_traintest[taski][-1].tolist(), method=method))
            container_pred[taski].append(get_embeddings(container_pred[taski][-1].tolist(), method=method))

    del train_w2v_model
    del word2vec_model
    del glove_model
if 5:
    from typing import Dict, List, Tuple, Sequence, Any
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    import seaborn as sns

    #generates a confusion matrix between hand labelled data and model predictions
    def getConfMatrix(pred_data, actual):
        conf_mat = confusion_matrix(actual, pred_data, labels=[0,1]) 
        micro = f1_score(actual, pred_data, average='micro') 
        macro = f1_score(actual,pred_data, average='macro')
        sns.heatmap(conf_mat, annot = True, fmt=".0f", annot_kws={"size": 18})
        print('F1 Micro: '+ str(micro))
        print('F1 Macro: '+ str(macro))

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.utils.data as data
    import math
    import copy, os

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import ray
    from ray import train, tune
    from ray.train import Checkpoint
    from ray.tune.schedulers import ASHAScheduler

    global_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class myRNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_layers=1):
            super().__init__()
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)
            out, _ = self.rnn(x, h0)
            out = self.fc(out[:, -1, :])
            return out

    class myLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_layers=1):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
            c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out

    class myGRU(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_layers=1, bidirectional=False):
            super().__init__()
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h0 = torch.zeros(self.gru.num_layers * 2 if self.gru.bidirectional else self.gru.num_layers, x.size(0), self.gru.hidden_size).to(x.device)
            out, _ = self.gru(x, h0)
            out = self.fc(out[:, -1, :])
            return out
        
    class MultiHeadAttention(nn.Module):
        
        """
        The init constructor checks whether the provided d_model is divisible by the number of heads (num_heads). 
        It sets up the necessary parameters and creates linear transformations for
        query(W_q), key(W_k) and output(W_o) projections
        """
        def __init__(self, d_model, num_heads):
            super(self).__init__()
            assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
            
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads
            
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)
            
        """
        The scaled_dot_product_attention function computes the scaled dot-product attention given the 
        query (Q), key (K), and value (V) matrices. It uses the scaled dot product formula, applies a mask if 
        provided, and computes the attention probabilities using the softmax function.
        """    
        def scaled_dot_product_attention(self, Q, K, V, mask=None):
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            if mask is not None:
                attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
            attn_probs = torch.softmax(attn_scores, dim=-1)
            output = torch.matmul(attn_probs, V)
            return output
        
        """
        The split_heads and combine_heads functions handle the splitting and combining of the attention heads.
        They reshape the input tensor to allow parallel processing of different attention heads.
        """
        def split_heads(self, x):
            batch_size, seq_length, d_model = x.size()
            return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
            
        def combine_heads(self, x):
            batch_size, _, seq_length, d_k = x.size()
            return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
        """
        The forward function takes input query (Q), key (K), and value (V) tensors, 
        applies linear transformations, splits them into multiple heads, performs scaled dot-product attention,
        combines the attention heads, and applies a final linear transformation.
        """    
        def forward(self, Q, K, V, mask=None):
            Q = self.split_heads(self.W_q(Q))
            K = self.split_heads(self.W_k(K))
            V = self.split_heads(self.W_v(V))
            
            attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
            output = self.W_o(self.combine_heads(attn_output))
            return output
    class PositionWiseFeedForward(nn.Module):
        """
        PositionWiseFeedForward module. It takes d_model as the input dimension and d_ff 
        as the hidden layer dimension. 
        Two linear layers (fc1 and fc2) are defined with ReLU activation in between.
        """
        def __init__(self, d_model, d_ff):
            super(self).__init__()
            self.fc1 = nn.Linear(d_model, d_ff)
            self.fc2 = nn.Linear(d_ff, d_model)
            self.relu = nn.ReLU()
            
        """
        The forward function takes an input tensor x, applies the first linear transformation (fc1), 
        applies the ReLU activation, and then applies the second linear transformation (fc2). 
        The output is the result of the second linear transformation.
        """
        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))
    class PositionalEncoding(nn.Module):
        """
        The constructor (__init__) initializes the PositionalEncoding module. 
        It takes d_model as the dimension of the model and max_seq_length as the maximum sequence length. 
        It computes the positional encoding matrix (pe) using sine and cosine functions.
        """
        def __init__(self, d_model, max_seq_length):
            super(self).__init__()
            
            pe = torch.zeros(max_seq_length, d_model)
            position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            
            self.register_buffer('pe', pe.unsqueeze(0))
        
        """
        The forward function takes an input tensor x and adds the positional encoding to it. 
        The positional encoding is truncated to match the length of the input sequence (x.size(1)).
        """    
        def forward(self, x):
            return x + self.pe[:, :x.size(1)]
    class EncoderLayer(nn.Module):
        
        """
        The constructor (__init__) initializes the EncoderLayer module. 
        It takes hyperparameters such as d_model (model dimension), num_heads (number of attention heads), 
        d_ff (dimension of the feedforward network), and dropout (dropout rate). 
        It creates instances of MultiHeadAttention, PositionWiseFeedForward, and nn.LayerNorm. 
        Dropout is also defined as a module.
        """
        def __init__(self, d_model, num_heads, d_ff, dropout):
            super(self).__init__()
            self.self_attn = MultiHeadAttention(d_model, num_heads)
            self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
            
        """
        The forward function takes an input tensor x and a mask. 
        It applies the self-attention mechanism (self.self_attn), adds the residual connection 
        with layer normalization, applies the position-wise feedforward network (self.feed_forward),
        and again adds the residual connection with layer normalization. 
        Dropout is applied at both the self-attention and feedforward stages.
        The mask parameter is used to mask certain positions during the self-attention step, 
        typically to prevent attending to future positions in a sequence.
        """
        def forward(self, x, mask):
            attn_output = self.self_attn(x, x, x, mask)
            x = self.norm1(x + self.dropout(attn_output))
            ff_output = self.feed_forward(x)
            x = self.norm2(x + self.dropout(ff_output))
            return x
    class EncoderOnlyTransformer(nn.Module):
        """
        The constructor (__init__) initializes the Transformer module. 
        It takes several hyperparameters, including vocabulary sizes for the source and target languages 
        (src_vocab_size and tgt_vocab_size), model dimension (d_model), number of attention heads (num_heads), 
        number of layers (num_layers), dimension of the feedforward network (d_ff), maximum sequence length 
        (max_seq_length), and dropout rate (dropout).
        It sets up embeddings for both the encoder and decoder (encoder_embedding and decoder_embedding), 
        a positional encoding module (positional_encoding), encoder layers (encoder_layers), 
        decoder layers (decoder_layers), a linear layer (fc), and dropout.
        """
        
        def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
            super(self).__init__()
            self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
            self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
            self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

            self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

            self.fc = nn.Linear(d_model, tgt_vocab_size)
            self.dropout = nn.Dropout(dropout)

        """
        The generate_mask function creates masks for the source and target sequences. 
        It generates a source mask by checking if the source sequence elements are not equal to 0. 
        For the target sequence, it creates a mask by checking if the target sequence elements are not equal 
        to 0 and applies a no-peek mask to prevent attending to future positions.
        """
        def generate_mask(self, src, tgt):
            src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
            tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
            seq_length = tgt.size(1)
            nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
            tgt_mask = tgt_mask & nopeak_mask
            return src_mask, tgt_mask

        """
        The forward function takes source (src) and target (tgt) sequences as input. 
        It generates source and target masks using the generate_mask function. 
        The source and target embeddings are obtained by applying dropout to the positional embeddings of the 
        encoder and decoder embeddings, respectively. 
        The encoder layers are then applied to the source embeddings to get the encoder output (enc_output). 
        The decoder layers are applied to the target embeddings along with the encoder output, source mask, 
        and target mask to get the final decoder output (dec_output). The output is obtained by applying a linear layer to the decoder output.
        """
        def forward(self, src, tgt):
            src_mask, tgt_mask = self.generate_mask(src, tgt)
            src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))

            enc_output = src_embedded
            for enc_layer in self.encoder_layers:
                enc_output = enc_layer(enc_output, src_mask)

            output = self.fc(enc_output)
            return output

    def model_train(method: str, train_test_data: Sequence, train_test_label: Sequence, pred_data: Sequence) -> Any:
        if method == 'SVM':
            from sklearn.svm import SVC
            from sklearn.pipeline import Pipeline
            from sklearn.model_selection import train_test_split, KFold, GridSearchCV
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            

            X_train, X_test, y_train, y_test = train_test_split(train_test_data, train_test_label, test_size=0.2, random_state=42)
            the_pipe = Pipeline([
                ('estimator', SVC())
            ])
            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            param_grid = {
                'estimator__C': [0.1, 0.3, 0.5, 1, 2],
                'estimator__kernel': ['linear'],
                'estimator__gamma': ['auto']
            }

            grid_search = GridSearchCV(the_pipe, param_grid, cv=kf, n_jobs=-2, verbose=2, scoring='accuracy')
            grid_search.fit(X_train, y_train)

            y_pred = grid_search.predict(X_test)
            getConfMatrix(y_pred, y_test)

            y_output = grid_search.predict(pred_data)
            f1_score(y_test, y_pred, average='micro')

            s = pd.Series(y_output)
            return s
        
        elif method == 'ELM':
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            from skelm import ELMClassifier
            from sklearn.pipeline import Pipeline
            from sklearn.model_selection import train_test_split, KFold, GridSearchCV

            X_train, X_test, y_train, y_test = train_test_split(train_test_data, train_test_label, test_size=0.2, random_state=42)
            the_pipe = Pipeline([
                ('estimator', ELMClassifier())
            ]) 
            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            param_grid = {
                'estimator__alpha': [0.00000002],
            }
            grid_search = GridSearchCV(the_pipe, param_grid, cv=kf, n_jobs=-2, verbose=2, scoring='accuracy')
            grid_search.fit(X_train, y_train)

            y_pred = grid_search.predict(X_test)
            getConfMatrix(y_pred, y_test)

            y_output = grid_search.predict(pred_data)
            f1_score(y_test, y_pred, average='micro')

            s = pd.Series(y_output)
            return s
        
        elif method == 'GaussianProcess':
            from sklearn.pipeline import Pipeline
            from sklearn.model_selection import train_test_split, KFold, GridSearchCV
            from sklearn.gaussian_process import GaussianProcessClassifier
            from sklearn.gaussian_process.kernels import RBF
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            
            X_train, X_test, y_train, y_test = train_test_split(train_test_data, train_test_label, test_size=0.2, random_state=42)
            the_pipe = Pipeline([
                ('estimator', GaussianProcessClassifier())
            ])
            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            param_grid = {
                'estimator__random_state' : [42]
            }
            grid_search = GridSearchCV(the_pipe, param_grid, cv=kf, n_jobs=-2, verbose=2, scoring='accuracy')
            grid_search.fit(X_train, y_train)

            y_pred = grid_search.predict(X_test)
            getConfMatrix(y_pred, y_test)

            y_output = grid_search.predict(pred_data)
            f1_score(y_test, y_pred, average='micro')

            s = pd.Series(y_output)
            return s

        elif method == 'RNN':
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            from torch.utils.data import DataLoader, TensorDataset
            from torch.optim import Adam
            import torch
            from ray import train, tune
            from ray.tune.search.optuna import OptunaSearch
            import ray
            import optuna

            def train_RNN(config: Dict[str, float]):
                X_train, X_test, y_train, y_test = train_test_split(train_test_data['reviews'], train_test_data['sentiment'], test_size=0.2, random_state=42)
                X_train = torch.tensor(X_train.to_numpy()).to(global_device)
                y_train = torch.tensor(y_train.to_numpy()).to(global_device)
                X_test = torch.tensor(X_test.to_numpy()).to(global_device)
                y_test = torch.tensor(y_test.to_numpy()).to(global_device)
                train_dataset = TensorDataset(X_train, y_train)
                train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
                model = myRNN(input_size=X_train.shape[2], hidden_size=config['hidden_size'], output_size=1).to(global_device)
                criterion = nn.BCEWithLogitsLoss().to(global_device)
                optimizer = Adam(model.parameters(), lr=config['lr'])
                while True: # let train.RunConfig.stop determine when to stop, each iter is one epoch
                    for i, (X , y) in enumerate(train_loader):
                        optimizer.zero_grad()
                        y_pred = model(X.float())
                        loss = criterion(y_pred, y.float().view(-1, 1))
                        loss.backward()
                        optimizer.step()
                    y_pred = model(X_test.float())
                    y_pred = [1 if pred > 0.5 else 0 for pred in y_pred]
                    accuracy = accuracy_score(y_test, y_pred)
                    train.report({"acc_report": accuracy})
            search_space = {
                'hidden_size': tune.randint(50, 200),
                'lr': tune.loguniform(1e-4, 1e-1),
                'batch_size': tune.choice([32, 64, 128])
            }

            algo = OptunaSearch()
            tuner = tune.Tuner(
                train_RNN,
                tune_config=tune.TuneConfig(
                    num_samples=10,
                    metric='acc_report',
                    mode='max',
                    search_alg=algo
                ),
                run_config=train.RunConfig(
                    name='rnn_tuner',
                    storage_path=os.path.join(os.getcwd(),'log', 'optuna_storage'),
                    stop={"training_iteration": 100,
                        "acc_report": 0.95},
                ),
                param_space=search_space,
            )
            results = tuner.fit()
            print("Best config is:", results.get_best_result().config)

            # train agian with the best config
            pass

        elif method == 'LSTM':
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            from torch.utils.data import DataLoader, TensorDataset
            from torch.optim import Adam
            import torch
            from ray import train, tune
            from ray.tune.search.optuna import OptunaSearch
            import ray
            import optuna

            def train_LSTM(config: Dict[str, float]):
                X_train, X_test, y_train, y_test = train_test_split(train_test_data['reviews'], train_test_data['sentiment'], test_size=0.2, random_state=42)
                X_train = torch.tensor(X_train.to_numpy()).to(global_device)
                y_train = torch.tensor(y_train.to_numpy()).to(global_device)
                X_test = torch.tensor(X_test.to_numpy()).to(global_device)
                y_test = torch.tensor(y_test.to_numpy()).to(global_device)
                train_dataset = TensorDataset(X_train, y_train)
                train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
                model = myLSTM(input_size=X_train.shape[2], hidden_size=config['hidden_size'], output_size=1).to(global_device)
                criterion = nn.BCEWithLogitsLoss().to(global_device)
                optimizer = Adam(model.parameters(), lr=config['lr'])
                while True: # let train.RunConfig.stop determine when to stop, each iter is one epoch
                    for i, (X , y) in enumerate(train_loader): # one batch
                        optimizer.zero_grad()
                        y_pred = model(X.float())
                        loss = criterion(y_pred, y.float().view(-1, 1))
                        loss.backward()
                        optimizer.step()
                    y_pred = model(X_test.float())
                    y_pred = [1 if pred > 0.5 else 0 for pred in y_pred]
                    accuracy = accuracy_score(y_test, y_pred)
                    train.report({"acc_report": accuracy})
            search_space = {
                'hidden_size': tune.randint(50, 200),
                'lr': tune.loguniform(1e-4, 1e-1),
                'batch_size': tune.choice([32, 64, 128])
            }

            algo = OptunaSearch()
            tuner = tune.Tuner(
                train_LSTM,
                tune_config=tune.TuneConfig(
                    num_samples=10,
                    metric='acc_report',
                    mode='max',
                    search_alg=algo
                ),
                run_config=train.RunConfig(
                    name='lstm_tuner',
                    storage_path=os.path.join(os.getcwd(),'log', 'optuna_storage'),
                    stop={"training_iteration": 333,
                        "acc_report": 0.95},
                ),
                param_space=search_space,
            )
            results = tuner.fit()
            print("Best config is:", results.get_best_result().config)

            # train agian with the best config
            pass

        elif method == 'GRU':
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            from torch.utils.data import DataLoader, TensorDataset
            from torch.optim import Adam
            import torch
            from ray import train, tune
            from ray.tune.search.optuna import OptunaSearch
            import ray
            import optuna

            def train_GRU(config: Dict[str, float]):
                X_train, X_test, y_train, y_test = train_test_split(train_test_data['reviews'], train_test_data['sentiment'], test_size=0.2, random_state=42)
                X_train = torch.tensor(X_train.to_numpy()).to(global_device)
                y_train = torch.tensor(y_train.to_numpy()).to(global_device)
                X_test = torch.tensor(X_test.to_numpy()).to(global_device)
                y_test = torch.tensor(y_test.to_numpy()).to(global_device)
                train_dataset = TensorDataset(X_train, y_train)
                train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
                model = myGRU(input_size=X_train.shape[2], hidden_size=config['hidden_size'], output_size=1, bidirectional=config['bidirectional']).to(global_device)
                criterion = nn.BCEWithLogitsLoss().to(global_device)
                optimizer = Adam(model.parameters(), lr=config['lr'])
                while True: # let train.RunConfig.stop determine when to stop, each iter is one epoch
                    for i, (X , y) in enumerate(train_loader): # one batch
                        optimizer.zero_grad()
                        y_pred = model(X.float())
                        loss = criterion(y_pred, y.float().view(-1, 1))
                        loss.backward()
                        optimizer.step()
                    y_pred = model(X_test.float())
                    y_pred = [1 if pred > 0.5 else 0 for pred in y_pred]
                    accuracy = accuracy_score(y_test, y_pred)
                    train.report({"acc_report": accuracy})
            search_space = {
                'hidden_size': tune.randint(50, 200),
                'lr': tune.loguniform(1e-4, 1e-1),
                'batch_size': tune.choice([32, 64, 128]),
                'bidirectional': tune.choice([True, False])
            }

            algo = OptunaSearch()
            tuner = tune.Tuner(
                train_GRU,
                tune_config=tune.TuneConfig(
                    num_samples=10,
                    metric='acc_report',
                    mode='max',
                    search_alg=algo
                ),
                run_config=train.RunConfig(
                    name='gru_tuner',
                    storage_path=os.path.join(os.getcwd(),'log', 'optuna_storage'),
                    stop={"training_iteration": 333,
                        "acc_report": 0.95},
                ),
                param_space=search_space,
            )
            results = tuner.fit()
            print("Best config is:", results.get_best_result().config)

            # train agian with the best config
            pass

        elif method == 'BERT' or method == "DistilBert" or method == "Roberta":
            if method == 'BERT':
                PRETRAIN_HF_NAME = 'bert-base-cased'
                FINETUNED = 'bert'
            elif method == 'DistilBert':
                PRETRAIN_HF_NAME = 'distilbert-base-cased'
                FINETUNED = 'distilbert'
            elif method == 'Roberta':
                PRETRAIN_HF_NAME = 'roberta-base'
                FINETUNED = 'roberta'
            
            
        elif method == 'costumized':
            raise NotImplementedError()
        else:
            raise ValueError('method not supported')
        

    # if we use ML based model, we literally fall back to BOW.
    def sum_up_embeddings(corpus) -> list[np.ndarray]:
        summed_arrays = [np.sum(sentences, axis=0) for sentences in corpus]
        return summed_arrays

    for taski in range(len(taskflows)):

        
        if taskflows[taski] & F.MODEL_SVM:
            train_test_data = sum_up_embeddings(container_traintest[taski][-1])
            train_test_label = container_traintest[taski][0]['sentiment'].tolist()
            pred_data = sum_up_embeddings(container_pred[taski][-1])
            result = model_train('SVM', train_test_data, train_test_label, pred_data)
            output: pd.DataFrame = container_pred[taski][0]
            output['sentiment'] = result
            output.to_csv(os.path.join('submit', 'out', 'SVM.csv'), index=False)
        elif taskflows[taski] & F.MODEL_ELM:
            train_test_data = sum_up_embeddings(container_traintest[taski][-1])
            train_test_label = container_traintest[taski][0]['sentiment'].tolist()
            pred_data = sum_up_embeddings(container_pred[taski][-1])
            result = model_train('ELM', train_test_data, train_test_label, pred_data)
            output: pd.DataFrame = container_pred[taski][0]
            output['sentiment'] = result
            output.to_csv(os.path.join('submit', 'out', 'ELM.csv'), index=False)
        elif taskflows[taski] & F.MODEL_GP:
            train_test_data = sum_up_embeddings(container_traintest[taski][-1])
            train_test_label = container_traintest[taski][0]['sentiment'].tolist()
            pred_data = sum_up_embeddings(container_pred[taski][-1])
            result = model_train('GaussianProcess', train_test_data, train_test_label, pred_data)
            output: pd.DataFrame = container_pred[taski][0]
            output['sentiment'] = result
            output.to_csv(os.path.join('submit', 'out', 'GaussianProcess.csv'), index=False)
        elif taskflows[taski] & F.MODEL_RNN:
            container_traintest[taski].append(model_train('RNN', container_traintest[taski][0], container_pred[taski][0]))
        elif taskflows[taski] & F.MODEL_LSTM:
            container_traintest[taski].append(model_train('LSTM', container_traintest[taski][0], container_pred[taski][0]))
        elif taskflows[taski] & F.MODEL_GRU:
            container_traintest[taski].append(model_train('GRU', container_traintest[taski][0], container_pred[taski][0]))
        elif taskflows[taski] & F.MODEL_BERT:
            container_traintest[taski].append(model_train('BERT', container_traintest[taski][0], container_pred[taski][0]))
        elif taskflows[taski] & F.MODEL_DISTILBERT:
            container_traintest[taski].append(model_train('DistilBert', container_traintest[taski][0], container_pred[taski][0]))
        elif taskflows[taski] & F.MODEL_ROBERTA:
            container_traintest[taski].append(model_train('Roberta', container_traintest[taski][0], container_pred[taski][0]))
        elif taskflows[taski] & F.MODEL_COSTUMIZED:
            container_traintest[taski].append(model_train('costumized', container_traintest[taski][0], container_pred[taski][0]))
        else:
            pass
