import os, sys, re, math, glob
import datetime
import stanfordnlp
import numpy as np
from tqdm import tqdm
from collections import OrderedDict, Counter, defaultdict
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.metrics import mean_squared_error

from models.Paper import Paper
from models.Review import Review
from models.ScienceParseReader import ScienceParseReader

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def clean_text(input):
    cleaned = input.strip()
    cleaned = re.sub("\n([0-9]*\n)+", "\n", cleaned)
    return cleaned

def preprocess(input, only_char=False, lower=False, stop_remove=False, stemming=False):
    #input = re.sub(r'[^\x00-\x7F]+',' ', input)
    if lower: input = input.lower()
    if only_char:
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(input)
        input = ' '.join(tokens)
    tokens = word_tokenize(input)
    if stop_remove:
        tokens = [w for w in tokens if not w in stopwords.words('english')]

    # also remove one-length word
    tokens = [w for w in tokens if len(w) > 1]
    return " ".join(tokens)

def evaluate(y, y_):
    return math.sqrt(mean_squared_error(y, y_))

def choose_label(x,y, size=5, label=False):

    # [size x 9]
    y = np.array(y)

    # (1) only choose label
    if label is not False and label >= 0:
        y = y[:,[label]]

    # (2) remove None/Nan examples
    x = x[~np.isnan(y).flatten()]
    y = y[~np.isnan(y)]
    y = np.reshape(y, (-1,1))

    assert x.shape[0] == y.shape[0]

    mean_aspects = []
    major_aspects = []
    evaluate_mean = []
    evaluate_major = []
    for aid, y_aspect in enumerate(y.T):
        #import pdb; pdb.set_trace()
        mean_aspect = np.average(y_aspect)
        #y_aspect_int = [int(yone) for yone in y_aspect]
        major_aspect = Counter(y_aspect).most_common(1)[0][0]
        mean_aspects.append(mean_aspect)
        major_aspects.append(major_aspect)

        evaluate_mean_aspect = evaluate(y_aspect, [mean_aspect] * len(y_aspect))
        evaluate_major_aspect = evaluate(y_aspect, [major_aspect] * len(y_aspect))
        #print aid,evaluate_mean_aspect, evaluate_major_aspect
        evaluate_mean.append(evaluate_mean_aspect)
        evaluate_major.append(evaluate_major_aspect)

    return x,y, evaluate_mean, evaluate_major, mean_aspects, major_aspects

def prepare_data(
    data_dir,
    vocab_path='vocab',
    max_vocab_size = 20000,
    max_len=1000,
    split_sentence=True):
    
    data_type = data_dir.split('/')[-1]
    vocab_path += '.' + data_type
    if max_vocab_size: vocab_path += '.'+str(max_vocab_size)
    vocab_path = data_dir +'/'+ vocab_path
    
    label_scale = 5
    if 'iclr' in data_dir.lower():
        fill_missing = False
        aspects = ['RECOMMENDATION', 'SUBSTANCE', 'APPROPRIATENESS','MEANINGFUL_COMPARISON','SOUNDNESS_CORRECTNESS','ORIGINALITY','CLARITY', 'IMPACT', 'RECOMMENDATION_ORIGINAL']
        review_dir_postfix = '' #'_annotated'
    elif 'acl' in data_dir.lower():
        fill_missing = True
        aspects = ['RECOMMENDATION', 'SUBSTANCE', 'APPROPRIATENESS','MEANINGFUL_COMPARISON','SOUNDNESS_CORRECTNESS','ORIGINALITY','CLARITY','IMPACT', 'REVIEWER_CONFIDENCE' ]
        review_dir_postfix = ''
    else:
        print('wrong dataset:',data_dir)
        sys.exit(1)

    if split_sentence:
        tokenizer = stanfordnlp.Pipeline(processors='tokenize',use_gpu=False)
        
    datasets = ['train', 'dev', 'test']
    data = []

    for dataset in datasets:

        review_dir = os.path.join(data_dir, dataset, 'reviews%s/'%(review_dir_postfix))
        scienceparse_dir = os.path.join(data_dir, dataset, 'parsed_pdfs/')

        paper_json_filenames = sorted(glob.glob('{}/*.json'.format(review_dir)))

        for paper_json_filename in tqdm(paper_json_filenames):

            d = {}

            paper = Paper.from_json(paper_json_filename)
            paper.SCIENCEPARSE = ScienceParseReader.read_science_parse(paper.ID, paper.TITLE, paper.ABSTRACT, scienceparse_dir)
            
            reviews = []
            for review in paper.REVIEWS:
                reviews.append(review)

            sections = paper.SCIENCEPARSE.get_sections_dict()

            for section_name, section_content in sections.items():
                if 'introduction' in section_name.lower():
                    cleanded_text = clean_text(section_content.lower())
                    if cleanded_text:
                        if split_sentence:
                            tokenized = tokenizer(cleanded_text)
                            section_content = [' '.join([word.text.lower() for word in sent.words]) for sent in tokenized.sentences]
                            section_content = [preprocess(sentence, only_char=False, lower=True, stop_remove=False, stemming=False) for sentence in section_content]
                            section_content = [sentence for sentence in section_content if len(sentence) > 0]
                        else:
                            section_content = cleanded_text
                        d['paper_content'] = section_content
                continue
                
            d['reviews'] = reviews
            
            if 'paper_content' in d.keys():
                data.append(d)

    print('Total number of papers %d' %(len(data)))
    print('Total number of reviews %d' %(np.sum([len(r['reviews']) for r in data ])))
            
    # Loading DATA
    print('Reading reviews from...')
    data_padded = []

    x_paper = [] #[None] * len(reviews)
    y = [] #[None] * len(reviews)
    for d in data:

        paper_content = d['paper_content']
        reviews = d['reviews']

        xone = paper_content
        yone = [np.nan] * len(aspects)

        for aid, aspect in enumerate(aspects):
            aspect_score = []   
            for rid, review in enumerate(reviews):
                if aspect in review.__dict__ and review.__dict__[aspect] is not None:
                    aspect_score.append(float(review.__dict__[aspect]))
            if aspect_score:
                yone[aid] = np.mean(aspect_score)

        x_paper.append(xone)
        y.append(yone)

    x_paper = np.array(x_paper)
    y = np.array(y, dtype=np.float32)

    # add average value of missing aspect value
    if fill_missing:
        col_mean = np.nanmean(y,axis=0)
        inds = np.where(np.isnan(y))
        y[inds] = np.take(col_mean, inds[1])

    print('Total dataset: %d'%(len(x_paper)),x_paper.shape, y.shape)
    data_padded.append(x_paper)
    data_padded.append(y)

    return data_padded,label_scale,aspects

def prepare_data2(
    data_dir,
    vocab_path='vocab',
    max_vocab_size = 20000,
    max_len=1000,
    split_sentence=True):
    
    data_type = data_dir.split('/')[-1]
    vocab_path += '.' + data_type
    if max_vocab_size: vocab_path += '.'+str(max_vocab_size)
    vocab_path = data_dir +'/'+ vocab_path
    
    label_scale = 5
    if 'iclr' in data_dir.lower():
        fill_missing = False
        aspects = ['RECOMMENDATION', 'SUBSTANCE', 'APPROPRIATENESS','MEANINGFUL_COMPARISON','SOUNDNESS_CORRECTNESS','ORIGINALITY','CLARITY', 'IMPACT', 'RECOMMENDATION_ORIGINAL']
        review_dir_postfix = '' #'_annotated'
    elif 'acl' in data_dir.lower():
        fill_missing = True
        aspects = ['RECOMMENDATION', 'SUBSTANCE', 'APPROPRIATENESS','MEANINGFUL_COMPARISON','SOUNDNESS_CORRECTNESS','ORIGINALITY','CLARITY','IMPACT', 'REVIEWER_CONFIDENCE' ]
        review_dir_postfix = ''
    else:
        print('wrong dataset:',data_dir)
        sys.exit(1)

    if split_sentence:
        tokenizer = stanfordnlp.Pipeline(processors='tokenize',use_gpu=False)
        
    datasets = ['train', 'dev', 'test']
    data = defaultdict(list)

    for dataset in datasets:

        review_dir = os.path.join(data_dir, dataset, 'reviews%s/'%(review_dir_postfix))
        scienceparse_dir = os.path.join(data_dir, dataset, 'parsed_pdfs/')

        paper_json_filenames = sorted(glob.glob('{}/*.json'.format(review_dir)))

        for paper_json_filename in tqdm(paper_json_filenames):

            d = {}

            paper = Paper.from_json(paper_json_filename)
            paper.SCIENCEPARSE = ScienceParseReader.read_science_parse(paper.ID, paper.TITLE, paper.ABSTRACT, scienceparse_dir)
            
            reviews = []
            for review in paper.REVIEWS:
                reviews.append(review)

            sections = paper.SCIENCEPARSE.get_sections_dict()

            for section_name, section_content in sections.items():
                if 'introduction' in section_name.lower():
                    cleanded_text = clean_text(section_content.lower())
                    if cleanded_text:
                        if split_sentence:
                            tokenized = tokenizer(cleanded_text)
                            section_content = [' '.join([word.text.lower() for word in sent.words]) for sent in tokenized.sentences]
                            section_content = [preprocess(sentence, only_char=False, lower=True, stop_remove=False, stemming=False) for sentence in section_content]
                            section_content = [sentence for sentence in section_content if len(sentence) > 0]
                        else:
                            section_content = cleanded_text
                        d['paper_content'] = section_content
                continue
                
            d['reviews'] = reviews
            
            if 'paper_content' in d.keys():
                data[dataset].append(d)

    print('Total number of papers %d' %(np.sum([len(d) for _,d in data.items()])))
    print('Total number of reviews %d' %(np.sum([len(r['reviews']) for _,d in data.items() for r in d ])))
            
    # Loading DATA
    print('Reading reviews from...')
    data_padded = []

    for dataset in datasets:
        
        x_paper = [] #[None] * len(reviews)
        y = [] #[None] * len(reviews)
        
        for d in data[dataset]:

            paper_content = d['paper_content']
            reviews = d['reviews']

            xone = paper_content
            yone = [np.nan] * len(aspects)

            for aid, aspect in enumerate(aspects):
                aspect_score = []   
                for rid, review in enumerate(reviews):
                    if aspect in review.__dict__ and review.__dict__[aspect] is not None:
                        aspect_score.append(float(review.__dict__[aspect]))
                if aspect_score:
                    yone[aid] = np.mean(aspect_score)

            x_paper.append(xone)
            y.append(yone)

        x_paper = np.array(x_paper)
        y = np.array(y, dtype=np.float32)

        # add average value of missing aspect value
        if fill_missing:
            col_mean = np.nanmean(y,axis=0)
            inds = np.where(np.isnan(y))
            y[inds] = np.take(col_mean, inds[1])

        print('Total dataset: %d'%(len(x_paper)),x_paper.shape, y.shape)
        data_padded.append(x_paper)
        data_padded.append(y)

    return data_padded,label_scale,aspects