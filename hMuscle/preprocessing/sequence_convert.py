import numpy as np
import sys
import os

# =========================================================
# [SETTINGS] File Paths
# =========================================================
INPUT_FILE = '../data/top30k_isoforms.pep'
OUTPUT_SEQ = '../results/amino_seq/protein_sequences.npy'
OUTPUT_ID  = '../results/amino_seq/protein_ids.npy'
TRUNCATION_LENGTH = 6000 

# =========================================================
# [SETTINGS] Amino Acid Map
# =========================================================
aa_num_dict = {
    'F': 1, 'L': 2, 'I': 3, 'M': 4, 'V': 5, 'S': 6, 'P': 7, 'T': 8, 'A': 9,
    'Y': 10, 'H': 11, 'Q': 12, 'N': 13, 'K': 14, 'D': 15, 'E': 16, 'C': 17,
    'W': 18, 'R': 19, 'G': 20
}
# Add special chars safely
for k in ['*', 'X', 'U', 'Z', 'B']:
    aa_num_dict[k] = 0

def pad_sequence(seq_list, max_len):
    padded = np.zeros((len(seq_list), max_len), dtype=np.int32)
    for i, seq in enumerate(seq_list):
        length = min(len(seq), max_len)
        padded[i, :length] = seq[:length]
    return padded

def main():
    if not os.path.exists(INPUT_FILE):
        print "Error: Input file not found -> " + INPUT_FILE
        sys.exit(1)

    print "Reading: " + INPUT_FILE + " ..."
    
    seq_data = []
    id_list = []
    
    current_id = None
    current_seq = []

    try:
        with open(INPUT_FILE, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line: continue

            if line.startswith('>'):
                if current_id is not None and len(current_seq) >= 3:
                    numseq = []
                    for j in range(len(current_seq) - 2):
                        try:
                            n1 = aa_num_dict.get(current_seq[j], 0)
                            n2 = aa_num_dict.get(current_seq[j+1], 0)
                            n3 = aa_num_dict.get(current_seq[j+2], 0)
                            
                            if n1 > 0 and n2 > 0 and n3 > 0:
                                ngram = (n1 - 1) * 400 + (n2 - 1) * 20 + n3
                                numseq.append(ngram)
                        except:
                            pass
                    
                    if len(numseq) > 0:
                        seq_data.append(numseq)
                        id_list.append(current_id)

                current_id = line.split()[0].replace('>', '')
                current_seq = []
            else:
                current_seq.extend(list(line))

        # Last sequence
        if current_id is not None and len(current_seq) >= 3:
            numseq = []
            for j in range(len(current_seq) - 2):
                n1 = aa_num_dict.get(current_seq[j], 0)
                n2 = aa_num_dict.get(current_seq[j+1], 0)
                n3 = aa_num_dict.get(current_seq[j+2], 0)
                if n1 > 0 and n2 > 0 and n3 > 0:
                    ngram = (n1 - 1) * 400 + (n2 - 1) * 20 + n3
                    numseq.append(ngram)
            
            if len(numseq) > 0:
                seq_data.append(numseq)
                id_list.append(current_id)

        print "Conversion Complete. Total sequences: " + str(len(seq_data))
        
        if len(seq_data) > 0:
            X_np = pad_sequence(seq_data, TRUNCATION_LENGTH)
            ids_np = np.array(id_list)
            
            print "Data Shape: " + str(X_np.shape)
            
            np.save(OUTPUT_SEQ, X_np)
            np.save(OUTPUT_ID, ids_np)
            
            print "Saved to: " + OUTPUT_SEQ
            print "Saved to: " + OUTPUT_ID
        else:
            print "No valid sequences found."

    except Exception as e:
        print "An error occurred:"
        print str(e)

if __name__ == "__main__":
    main()
