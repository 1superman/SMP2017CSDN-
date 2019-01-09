# -*- coding: utf-8 -*-
import os
import math
import xgboost as xgb
import pandas as pd
import numpy as np
import multiprocessing
import logging  
import jieba 
import jieba.analyse
import jieba.posseg as pseg
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing  
import gensim
from gensim import corpora  
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from collections import Counter, defaultdict 
logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)  
stop = open('stopwords.txt').read()

try:
    os.mkdir('博客文本')
    os.mkdir('博客词向量')
    os.mkdir('model')
    os.mkdir('featurescore')
except:
    pass
    
def Add_word():
    global t1, labelspace_dic
    t1 = pd.read_table('train1.txt', header=None, sep='\x01', names=['contentid', 'k1','k2','k3','k4','k5'])
    labelspace_dic = Counter(list(t1['k1'])+list(t1['k2'])+list(t1['k3'])+list(t1['k4'])+list(t1['k5']))
    labelspace = list(set(list(t1['k1'])+list(t1['k2'])+list(t1['k3'])+list(t1['k4'])+list(t1['k5'])))
    for i in labelspace:
        if str(i) != 'nan':
            jieba.add_word(i)
    
def cut_word(line):
    ss = ''
    try:
        line = pseg.cut(line.strip())
        for i in line:
            if (i.flag == 'n' or i.flag == 'eng') and (i.word.__len__() > 1):
                ss += i.word.lower()
                ss += ':'
                ss += i.flag
                ss += '\x01'
        return ss
    except:
        return np.nan
    
def Read_content():
    csv_path = 'train1_text.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv('train1_text.csv')
    else:
        df = open('1_BlogContent.txt', encoding='utf8').readlines()
        df = pd.DataFrame(df, columns=['data'])
        df['data'] = df['data'].str.split('\001')
        df['contentid'] = df['data'].apply(lambda x:x[0])
        df = t1.merge(df, on='contentid', how='left')
        df.to_csv('train1_text.csv', index=False, encoding='utf8')
    return df

def Pseg_dic(df):
    pseg_dic = {}
    for i in df['cut_word_text']:
        for j in i[:-1]:
            a, b = j.split(':')
            pseg_dic[a] = b
    for i in df['cut_word_title']:
        for j in i[:-1]:
            a, b = j.split(':')
            pseg_dic[a] = b       
    return pseg_dic

def Cancel_stop(line):
    new = []
    for i in line:
        if i == '':
            continue
        if i not in stop:
            new.append(i)
    return ' '.join(new)

def Pre_tfidf(df):
    df['data'] = df['data'].map(eval)
    df['title'] = df['data'].apply(lambda x:x[1]) #提取title
    df['title'] = df['title'].map(cut_word) #title分词
    df['text'] = df['data'].apply(lambda x:x[2]) #提取内容
    df['text'] = df['text'].map(cut_word) #内容分词
    df['cut_word_text'] = df['text'].str.split('\x01')
    df['cut_word_title'] = df['title'].str.split('\x01')
    del df['data']
    pseg_dic = Pseg_dic(df) #构建词性字典
    df['cut_word_text'] = df['cut_word_text'].apply(lambda x:[i.split(':')[0] for i in x]) #分词整理
    df['cut_word_text'] = df['cut_word_text'].map(Cancel_stop)
    df['cut_word_title'] = df['cut_word_title'].apply(lambda x:[i.split(':')[0] for i in x]) #分词整理
    df['cut_word_title'] = df['cut_word_title'].map(Cancel_stop)
    return pseg_dic, df


def Tf(line):
    line = line.split(' ')[:-1]
    dd = dict(Counter(line))  
    c = len(line)
    tf = {}
    for i in dd.keys():
        tf[i] = dd[i]/c
    return tf

def Idf(df):
    idf = defaultdict(int)
    for i in df['cut_word_text']:
        for k in pseg_dic.keys():
            if k in i:
                idf[k] += 1
    idf = dict(idf)
    for i in idf.keys():
        if idf[i] == 1:
            idf[i] = 0
            continue
        idf[i] = len(df) / idf[i]
    log_idf = {}
    for i in idf.keys():
        if idf[i] != 0:
            log_idf[i] = idf[i]+1
    return log_idf

def Tf_idf(line):
    tfidf = {}
    for i in line.keys():
        try:
            tfidf[i] = line[i] * math.log(idf[i])
        except:
            pass
    return tfidf

def First_place(x):
    try:
        return x[1].split(' ').index(x[0])/x[1].split(' ').__len__()
    except:
        return -1
       
def Last_place(x):
    try:
        return (x[1].split(' ').__len__() - x[1].split(' ')[::-1].index(x[0]))/x[1].split(' ').__len__()
    except:
        return -1
    
def Generate_sample(df, n):
    csv_path = 'task1_basic_feature.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df['tf'] = df['cut_word_text']+' '+df['cut_word_title']
        df['tf'] = df['tf'].map(Tf)    
        df['tfidf'] = df['tf'].map(Tf_idf)
        df['TOP10_tfidf'] = df['tfidf'].apply(lambda x:sorted(x.items(), key=lambda y:y[1], reverse=True)[:n])
        Top10 = list(df['TOP10_tfidf'])
        zz = []; tt = []
        df = df.sort_values('contentid')
        contentid = df['contentid']
        for i in range(len(contentid)):
            c = 0
            for j in range(n):
                try:
                    zz.append(Top10[i][j])
                    c += 1
                except:
                    pass
            tt.append([contentid[i], c])
        dfi = df
        for i in tt:
            try:
                for j in range(i[1]-1):
                    df = df.append(dfi.loc[dfi['contentid'] == i[0]])
            except:
                pass
        df = df.sort_values('contentid')
        df.index = range(len(df))
        zz = list(zip(*zz))
        df['key_word'], df['tfidf'] = zz
        df['tf'] = list(zip(df['tf'], df['key_word']))
        df['tf'] = df['tf'].apply(lambda x:x[0][x[1]])
        df['idf'] = df['key_word'].map(idf).map(math.log)
        df['tf_rank'] = df.groupby('contentid')['tf'].rank()
        df['idf_rank'] = df.groupby('contentid')['idf'].rank()
        df['tfidf_rank'] = df.groupby('contentid')['tfidf'].rank()
        df['keys'] = list(zip(df['k1'], df['k2'], df['k3'], df['k4'], df['k5']))
        df['label'] = list(zip(df['keys'], df['key_word']))
        df['label'] = df['label'].apply(lambda x: 1 if x[1] in x[0] else 0)
        df['Attributes'] = df['key_word'].map(pseg_dic)
        le = preprocessing.LabelEncoder()  
        le.fit(df['Attributes'])      
        df['Attributes'] = le.transform(df['Attributes'])
        le.fit(df['contentid'])
        df['text_num'] = le.transform(df['contentid'])
        df['len'] = df['key_word'].map(len)
        df['is_digit'] = df['key_word'].apply(lambda x: x.isdigit()).map({True:1,False:0})
        df['is_alpha'] = df['key_word'].apply(lambda x: x.isalpha()).map({True:1,False:0})
        df['digit_and_alpha'] = df['key_word'].apply(lambda x: not (x.isalpha() or x.isdigit()) and x.isalnum()).map({True:1,False:0})
        df['is_title'] = list(zip(df['key_word'], df['cut_word_title']))
        df['is_title'] = df['is_title'].apply(lambda x: 1 if x[0] in x[1] else 0)
        df['place'] = list(zip(df['key_word'], df['cut_word_text']))
        df['first_place'] = df['place'].map(First_place)
        df['last_place'] = df['place'].map(Last_place)
        df['len_text'] = df['cut_word_text'].str.split(' ').map(len)
        df = df.drop(['TOP10_tfidf', 'k1', 'k2', 'k3', 'k4', 'k5', 'keys', 'title', 'text', 'place'], axis=1)
        df.to_csv(csv_path, index=False, encoding='utf8')
    return df

def cos(line):   #余弦相似度
    vector1,vector2 = line
    dot_product = 0.0  
    normA = 0.0  
    normB = 0.0  
    for a,b in zip(vector1,vector2):  
        dot_product += a*b  
        normA += a**2  
        normB += b**2  
    if normA == 0.0 or normB==0.0:  
        return None  
    else:  
        return dot_product / ((normA*normB)**0.5) 

def Doc_vec(line):
    c = labelspace_dic[line[0]]
    doc = w2model[line[0]] * (c+1)      
    for i in line[1:]:
        if i == '':
            continue
        try:
            doc += w2model[i] * (labelspace_dic[i]+1)
            c += labelspace_dic[i]
        except:
            try:
                doc += w2model[i]
                c += 1
            except:
                pass
    doc /= c
    return doc.tolist()

def zz(line):
    new = []
    for i in line:
        if i != '':
            new.append(i)
    return new

def Word_vec(df):
    global w2model
    csv_path = 'word2vec.csv'
    model_path = 'word2vec.txt'
    if os.path.exists(csv_path):
        d = pd.read_csv(csv_path)
        w2model = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(model_path)
    else:
        d = df.drop_duplicates('contentid')
        d.dropna(inplace=True)
        word = d['cut_word_text'] + ' ' + d['cut_word_title']
        text = open('text.txt','w',encoding='utf8')
        for i in word:
            text.write(i)
            text.write('\n')
        text.close()
        w2model = Word2Vec(LineSentence('text.txt'), size=200, window=5, min_count=1, workers=multiprocessing.cpu_count())
        w2model.wv.save_word2vec_format(model_path)
        d['text'] = df['cut_word_title']+' '+df['cut_word_text']
        d['text'] = d['text'].str.split(' ').map(zz)
        d['doc2vec'] = d['text'].map(Doc_vec)
        doc = list(zip(*list(d['doc2vec'])))    
        for i in range(200):
            d['doc2vec'+str(i)] = doc[i]
        d = d[['contentid', 'doc2vec']+['doc2vec'+str(i) for i in range(200)]]
        d.to_csv(csv_path, index=False, encoding='utf8')
    return d, w2model

def Topic_model(model, df):
    csv_path = '{}.csv'.format(model)
    if os.path.exists(csv_path):
        dd = pd.read_csv(csv_path)
    else:
        model_path = 'model/{}.model'.format(model)
        texts = open('text.txt', encoding='utf8').readlines()
        texts = [i.split(' ') for i in texts]
        dictionary = corpora.Dictionary(texts)  
        corpus = [dictionary.doc2bow(text) for text in texts]  
        if model == 'lda':
            if os.path.exists(model_path):
                model = gensim.models.LdaModel.load(model_path)
            else:
                model = gensim.models.ldamodel.LdaModel(corpus, num_topics=42, id2word = dictionary, passes=20)  
                model.save(model_path)
        id2 = model.id2word
        doc_topic = model.get_document_topics(corpus,minimum_probability=False)
        doc_topic = [list(zip(*i))[1] for i in doc_topic]
        df1 = pd.DataFrame({'contentid': pd.unique(df['contentid'])})
        df1['doc_topics'] = doc_topic
        doc_topic = list(zip(*doc_topic))        
        df1['doc_label'] = df1['doc_topics'].apply(lambda x:np.argmax(x))
        key_word = list(df['key_word'])
        it = {}
        for i in id2.keys():
            it[id2[i]] = i
        d = []
        for i in key_word:
            dd = {}
            try:      
                for j in model.get_term_topics(it[i]):
                    dd[j[0]] = j[1]
                d.append(dd)
            except:
                d.append(dd)
        for i in range(len(doc_topic)):
            df1['doc_topic'+str(i)] = doc_topic[i]
        dd = pd.DataFrame(d)
        dd.fillna(0, inplace=True)
        dd.columns = ['word_topic'+str(i) for i in range(len(dd.columns))]   
        dd['word_topics'] = dd.values.tolist()             
        dd['contentid'] = df['contentid']
        dd = dd.merge(df1, on='contentid')
        dd.to_csv(csv_path, index=False, encoding='utf8')
    return dd 

#def HDP(df):
    
def train(model):
    csv_path = 'task1_feature_{}.csv'.format(model)
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        Add_word() # jieba分词词库添加专业名词
        df = Read_content() #匹配训练集
        global idf, pseg_dic
        pseg_dic, df = Pre_tfidf(df) #磁性字典
        idf = Idf(df) #词的idf值
        df = Generate_sample(df, 100)
        word2vec, w2model = Word_vec(df)
        topic_model = Topic_model(model, df)
        df = df.merge(word2vec, on='contentid')
        topic_model = topic_model[topic_model.contentid.isin(pd.unique(df.contentid))]
        del topic_model['contentid']
        topic_model.index = range(len(topic_model))
        df = pd.concat([df, topic_model], axis=1)
        df['keyword_vec'] = df['key_word'].apply(lambda x: w2model[x])
        try:
            df['doc2vec'] = df['doc2vec'].map(eval)
        except:
            pass
        df['w2cos'] = list(zip(df['keyword_vec'], df['doc2vec']))
        df['w2cos'] = df['w2cos'].map(cos)
        try:
            df.word_topics = df.word_topics.map(eval)
            df.doc_topics = df.doc_topics.map(eval)
        except:
            pass
        df['{}cos'.format(model)] = pd.Series(list(zip(df['word_topics'], df['doc_topics']))).map(cos)

        w2 = list(zip(*list(df['keyword_vec'])))
        for i in range(200):
            df['word2vec'+str(i)] = w2[i]
        df.to_csv(csv_path, index=False, encoding='utf8')
    return df

def Eval_metric(preds,dtrain):
    dtrain = pd.Series(dtrain.get_label())
    preds = pd.Series(preds)
    preds = preds[preds == dtrain]
    return 'loss', preds.sum() / dtrain.sum()

def xgboost(trainx, n, n_ter): 
    cid = pd.unique(trainx['contentid'])
    train_id = cid[:600]
    val_id = cid[600:900]
    test_id = cid[900:]
    target = 'label'   
    feature_name = [x for x in trainx.columns if x not in ['contentid','tf_rank', 'idf_rank', 'tfidf_rank', 'word_topics', 'doc_topics', 'doc_label', 'text_num', 'cut_word_text', 'digit_and_alpha','is_digit', 'Attributes', 'cut_word_title', 'keyword_vec', 'doc2vec', 'word2vec', 'key_word', 'last_place', 'label']]
    w_feature = ['w2cos'] + ['word2vec' + str(i) for i in range(200)] + ['doc2vec' + str(i) for i in range(200)]
    d_feature = ['ldacos'] + ['word_topic' + str(i) for i in range(42)] + ['doc_topic' + str(i) for i in range(42)]
#    feature_name = [x for x in feature_name if x not in w_feature+d_feature]
#    feature_name = w_feature
#    feature_name = d_feature
    feature_name = pd.read_csv('featurescore/feature_score_stac1.csv')['feature']
    train = trainx[trainx['contentid'].isin(train_id)]
    val = trainx[trainx['contentid'].isin(val_id)]
    test = trainx[trainx['contentid'].isin(test_id)]
    testid = test['contentid']
    test_keyword = test['key_word']
    y = train[target]
    valy = val[target]
    testy = test[target]
    train = train[feature_name]
    val = val[feature_name] 
    test = test[feature_name]
    dtrain = xgb.DMatrix(train, label = y, missing=np.nan)
    dval = xgb.DMatrix(val, label = valy, missing=np.nan)
    dtest = xgb.DMatrix(test, label = testy, missing=np.nan)  
    
    params = {
                'booster':'gbtree',
                'objective': 'reg:logistic',
                'eval_metric': 'error',
#                'num_class': 2,
                'max_depth':10,
                'gamma': 0.1,                
                'alpha': 1,
                'lambda': 1,
                'min_child_weight': 50,
                'scale_pos_weight': 1,
                'subsample':0.75 ,       
                'colsample_bytree':1.,
                'eta': n,
                'silent' : 0
            }   
        
    watchlist  = [(dtrain,'train'),(dval,'val')]
    num_round = n_ter  
    model = xgb.train(params, dtrain,  num_round,evals=watchlist, verbose_eval=1) 
    
    result = model.predict(dtest)
    result = pd.DataFrame({'contentid': testid,'key_word':test_keyword, 'score': result, 'valy': testy})
    result = result.sort_values(['contentid', 'score'])
    g = result.groupby('contentid')
    result = pd.DataFrame()
    for i in pd.unique(testid):
        gi = g.get_group(i)
        result = result.append(gi[-3:])
    result.index = range(len(result))
    score = result['valy'].mean()
    result.to_csv('result.csv', index=False)
    
    feature_score = model.get_fscore()
    feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
    fs = []
    for (key,value) in feature_score:
        fs.append("{0},{1}\n".format(key,value))
        
#    w2v = feature_name[feature_name['feature'].isin(['word2vec'+str(i) for i in range(200)])]
#    w2v_score = w2v['score'].sum()
    
    with open('featurescore/feature_score_stac1.csv','w') as f:
        f.writelines("feature,score\n")
        f.writelines(fs)
    return train, score

#df_lda = train('lda')
#train, score = xgboost(df_lda ,0.1, 100)
