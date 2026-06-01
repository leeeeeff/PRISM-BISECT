import numpy as np
import os  # [필수] 파일 경로 확인용

# ------------------------------------------------------------------------------
# [추가] ID 매핑 로드 함수 및 전역 변수 설정
# ------------------------------------------------------------------------------
def load_mapping_dict(mapping_file_path):
    mapping_dict = {}
    if not os.path.exists(mapping_file_path):
        print("Warning: Mapping file not found at " + mapping_file_path)
        return mapping_dict
    
    with open(mapping_file_path, 'r') as f:
        # [수정 1] 첫 줄(헤더) 건너뛰기
        next(f, None) 
        
        for line in f:
            # BioMart 파일은 보통 탭(\t)이나 공백으로 구분됨
            parts = line.strip().split() 
            
            # [수정 2] 사용자의 파일 구조:
            # 0: GeneID, 1: ID ver, 2: TranscriptID, 3: Trans ver, 4: Gene Name
            # 그래서 parts[4]를 가져와야 합니다.
            if len(parts) >= 5: 
                ensembl_id = parts[0]   # 예: ENSG00000210049
                symbol = parts[4]       # 예: MT-TF
                mapping_dict[ensembl_id] = symbol
                
    print("Loaded " + str(len(mapping_dict)) + " ID mappings.")
    return mapping_dict

# [중요] 경로가 실행 위치(model 폴더) 기준으로 맞는지 확인하세요.
MAPPING_DICT = load_mapping_dict('../data/raw_data/data/id_lists/ensembl_to_symbol.txt')


# ------------------------------------------------------------------------------
# Generate labels 함수
# ------------------------------------------------------------------------------
def generate_label(X_train_seq, X_train_dm, X_train_other_seq, X_train_other_dm, X_train_geneid, X_train_geneid_other, X_test_geneid, positive_Gene):
    gene_count = 0
    gene_index_dic = {}
    y_train = np.array([])
    y_test = np.array([])
    gene_index = []

    # 1. Main Train Loop
    last_gID = ''
    for gID in X_train_geneid:
        gID_str = gID.decode('utf-8') if isinstance(gID, bytes) else gID
        gID_base = gID_str.split('.')[0]
        mapped_id = MAPPING_DICT.get(gID_base, MAPPING_DICT.get(gID_str, gID_str)) # 매핑 적용

        if gID != last_gID:
            if mapped_id in positive_Gene:     # mapped_id로 비교
                y_train = np.hstack((y_train, np.ones(1)))
            else:
                y_train = np.hstack((y_train, np.zeros(1)))
            gene_count += 1
            gene_index_dic[gID] = gene_count
            gene_index.append(gene_count)
            last_gID = gID
        else:
            y_train = np.hstack((y_train, y_train[-1]))
            gene_index.append(gene_count)

    y_CRF_point = y_train
    crf_bag_index = gene_index
    crf_bag_index = np.array(crf_bag_index)

    # 2. Other Train Loop
    last_gID = ''
    negadd = 0
    add_index = np.array([])
    for i in range(len(X_train_geneid_other)):
        gID = X_train_geneid_other[i]
        gID_str = gID.decode('utf-8') if isinstance(gID, bytes) else gID
        gID_base = gID_str.split('.')[0]
        mapped_id = MAPPING_DICT.get(gID_base, MAPPING_DICT.get(gID_str, gID_str)) # 매핑 적용

        if gID != last_gID:
            if mapped_id in positive_Gene:     # mapped_id로 비교
                y_train = np.hstack((y_train, np.ones(1)))
                gene_count += 1
                gene_index_dic[gID] = gene_count
                gene_index.append(gene_count)
                add_index = np.hstack((add_index, np.array(i)))
                last_gID = gID
            else:
                negadd += 1
                y_train = np.hstack((y_train, np.zeros(1)))
                gene_count += 1
                gene_index_dic[gID] = gene_count
                gene_index.append(gene_count)
                add_index = np.hstack((add_index, np.array(i)))
                last_gID = gID
        else:
            y_train = np.hstack((y_train, y_train[-1]))
            gene_index.append(gene_count)
            add_index = np.hstack((add_index, np.array(i)))

    add_index = add_index.astype(np.int64)
    X_train_seq = np.vstack((X_train_seq, X_train_other_seq[add_index]))
    X_train_dm = np.vstack((X_train_dm, X_train_other_dm[add_index]))

    # 3. Test Loop
    last_gID = '' # (안전하게 초기화 한 번 더)
    for gID in X_test_geneid:
        gID_str = gID.decode('utf-8') if isinstance(gID, bytes) else gID
        gID_base = gID_str.split('.')[0]
        mapped_id = MAPPING_DICT.get(gID_base, MAPPING_DICT.get(gID_str, gID_str)) # 매핑 적용

        if gID != last_gID:
            if mapped_id in positive_Gene:     # mapped_id로 비교
                y_test = np.hstack((y_test, np.ones(1)))
            else:
                y_test = np.hstack((y_test, np.zeros(1)))
            gene_count += 1
            gene_index_dic[gID] = gene_count
            last_gID = gID
        else:
            y_test = np.hstack((y_test, y_test[-1]))

    y_CRF_point = np.hstack((y_CRF_point, y_test))
    crf_bag_index = np.hstack((crf_bag_index, -1 * np.ones(y_test.shape[0])))

    gene_index = np.array(gene_index)
    return y_train, y_test, y_CRF_point, crf_bag_index, gene_index, gene_count, X_train_seq, X_train_dm

# ------------------------------------------------------------------------------
# Upsample 함수 (기존 로직 유지)
# ------------------------------------------------------------------------------
def upsample(y_train, gene_index, gene_count, X_train_seq, X_train_dm, unused_flag):
    pos_neg_index = np.zeros(len(gene_index))
    for i in range(len(gene_index)):
        if y_train[i] == 0 and np.sum(y_train[np.where(gene_index == gene_index[i])]) == 0:
            pos_neg_index[i] = 0
        elif y_train[i] == 0 and np.sum(y_train[np.where(gene_index == gene_index[i])]) > 0:
            pos_neg_index[i] = 2
        elif y_train[i] == 1 and len(np.where(gene_index == gene_index[i])[0]) == 1:
            pos_neg_index[i] = 1
        elif y_train[i] == 1 and len(np.where(gene_index == gene_index[i])[0]) > 1:
            pos_neg_index[i] = 2

    positive_sig_index = np.where(pos_neg_index == 1)[0]
    positive_mig_index = np.where(pos_neg_index == 2)[0]
    positive_index = np.hstack((positive_sig_index, positive_mig_index))

    negtive_index = np.where(pos_neg_index == 0)
    unused_index = np.where(unused_flag == 0)
    negtive_unused = np.array(list(set(negtive_index[0]).intersection(set(unused_index[0]))))
    np.random.shuffle(negtive_unused)
    negtive_index = negtive_unused[0: 15000]

    choose_index = np.hstack((positive_index, negtive_index))
    unused_flag[choose_index] = 1

    numadd = int(negtive_index.shape[0] - positive_index.shape[0])
    if numadd > 0:
        idx = np.random.randint(0, positive_sig_index.shape[0], numadd)
        selected_idx = positive_sig_index
        label = 1
    else:
        idx = np.random.randint(0, negtive_index.shape[0], -numadd)
        selected_idx = negtive_index
        label = 0

    add = selected_idx[list(idx)]
    add = list(add)
    y_train_upsmp = np.hstack((y_train[choose_index], label * np.ones(len(add))))
    X_train_seq_upsmp = np.vstack((X_train_seq[choose_index], X_train_seq[add]))
    X_train_dm_upsmp = np.vstack((X_train_dm[choose_index], X_train_dm[add]))

    negtive_index = np.where(pos_neg_index == 0)
    unused_index = np.where(unused_flag == 0)
    negtive_unused = np.array(list(set(negtive_index[0]).intersection(set(unused_index[0]))))
    if len(negtive_unused) < 15000:
        unused_flag[:] = 0
        unused_flag[positive_index] = 1

    return X_train_seq_upsmp, X_train_dm_upsmp, y_train_upsmp, unused_flag

# ------------------------------------------------------------------------------
# Make Batch 함수 (기존 로직 유지)
# ------------------------------------------------------------------------------
def make_batch(seqX):
    gpl_dic = {}
    nonspace = np.sign(seqX)
    aalen = np.sum(nonspace, 1)
    idx = np.argsort(aalen)
    len_srt = aalen[idx]
    stidx = 0
    gp_n = 0
    maxlen = 1000
    add_length = maxlen * 2
    for i in range(seqX.shape[0]):
        if len_srt[i] > maxlen:
            gpl_dic[gp_n] = (stidx, i - 1, maxlen)
            gp_n += 1
            stidx = i
            maxlen += add_length
            add_length *= 2
    gpl_dic[gp_n] = (stidx, seqX.shape[0] - 1, maxlen)
    return idx, gpl_dic
