import argparse
import glob
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-i", dest="input_dir", type=str, required=True, help="Discourse Graphbank directory")
args = parser.parse_args()

#fix wrong tags
replace = {
    'ayyt': 'attr', 
    'exam': 'examp',
    'exv': 'expv',
    'elabdet': 'elab',
    'sme': 'same',
    'cont': 'contr',
    'parallel': 'par',
    'contrast': 'contr'}

def get_dict(text_id, sa_idx, sb_idx, sa_segment, sb_segment, tag):
    return {
        'text_id': text_id,
        'idx_a': sa_idx,
        'idx_b': sb_idx,
        'text_a': sa_segment,
        'text_b': sb_segment,
        'relation': tag,
    }

def get_anno(annotations):
    a_idxs, b_idxs, tags = [], [], []
    for annotation in annotations:
        annotation = annotation.split(' ')
        if annotation[0] == annotation[1] and annotation[2] == annotation[3]:
            a_idxs.append(int(annotation[0]))
            b_idxs.append(int(annotation[2]))
            tag = annotation[-1].split('-')[0]
            if tag in replace.keys():
                tag = replace[tag]
            tags.append(tag)
        assert len(a_idxs) == len(b_idxs)
        assert len(a_idxs) == len(tags)
    return zip(a_idxs, b_idxs, tags)

anno1_dir = args.input_dir + '/annotator1/'
anno2_dir = args.input_dir + '/annotator2/'

filenames1 = sorted(glob.glob('{}/*'.format(anno1_dir)))
filenames1 = [filename for filename in filenames1 if 'annotation' not in filename]
filenames2 = sorted(glob.glob('{}/*'.format(anno1_dir)))
filenames2 = [filename for filename in filenames2 if 'annotation' not in filename]

sym_rel = ['par', 'contr', 'same']

counter = {}
rows_list = []
match = 0
miss_match = 0
for filename1, filename2 in zip(filenames1, filenames2):
    with open(filename1, encoding="ISO-8859-1") as f:
        data = f.read()
    segments = [segment for segment in data.split('\n') if len(segment) > 0]
    with open(filename1+'-annotation', encoding="ISO-8859-1") as f:
        annotations1 = f.read()
    with open(filename2+'-annotation', encoding="ISO-8859-1") as f:
        annotations2 = f.read()
    annotations1 = annotations1.strip().split('\n')
    annotations2 = annotations2.strip().split('\n')
    anno_tags1 = get_anno(annotations1)
    anno_tags2 = get_anno(annotations2)
    anno_tags = []
#     print(list(anno_tags1))
#     print(list(anno_tags2))
#     sys.exit()
    
    for anno_tag in anno_tags1:
        if anno_tag in anno_tags2:
            anno_tags.append(anno_tag)
            match += 1
        else:
            miss_match += 1
            
    a_idxs, b_idxs, tags = list(zip(*anno_tags))      
    text_id = filename1.split('/')[-1]
    for s_idx in range(len(segments)-1):
        rel_flag = False
        for idx in range(len(a_idxs)):
            if (s_idx == a_idxs[idx] and s_idx+1 == b_idxs[idx]) or (s_idx == b_idxs[idx] and s_idx+1 == a_idxs[idx]):
                rows_list.append(get_dict(text_id, s_idx, s_idx+1, segments[s_idx], segments[s_idx+1], tags[idx]))
                rel_flag = True
        if not rel_flag:
            rows_list.append(get_dict(text_id, s_idx, s_idx+1, segments[s_idx], segments[s_idx+1], 'none'))

print("Annotators correlation:")                        
print("   Match: %d tags" % match)
print("   Miss match: %d tags" % miss_match)

df = pd.DataFrame(rows_list)
df.to_csv('sentence_pair_rel.csv')