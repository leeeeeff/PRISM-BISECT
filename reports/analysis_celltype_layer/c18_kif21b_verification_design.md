# C18(L4*) 세포에서 KIF21B Isoform Usage 직접 검증 설계

**목적**: Short-read에서 C18이 AD-enriched + KIF21B-high + RBFOX1-low임을 확인함.  
Long-read BAM에서 C18 세포의 KIF21B isoform usage를 직접 측정하여 **motor domain switch가 C18에서 특이적으로 발생하는지** 검증.

---

## 핵심 가설

> "C18 cluster에서 RBFOX1 발현이 낮기 때문에 KIF21B exon17 inclusion이 감소하고,  
> 이것이 motor domain 소실 isoform의 증가로 이어진다.  
> 이 switch는 C18에서 특이적으로 강하며, AD 샘플에서 더 두드러진다."

검증 가능 예측:
1. C18 세포에서 KIF21B CT isoform (transcript293004.chr1.nic) usage가 다른 L4 cluster보다 낮다
2. C18 AD 샘플에서 KIF21B AD isoform (KIF21B-203) usage가 C18 CT 샘플보다 높다
3. C18의 KIF21B isoform ratio가 RBFOX1 발현량과 양의 상관 (r > 0)

---

## 데이터 가용성 확인

```
Long-read BAM (cell barcode 포함):
  위치: /home/dhkim1674/Project_AD_with_refTSS_novel/02_BAM/ (추정)
  형식: 10x Chromium compatible — CB (cell barcode), UB (UMI) BAM tags
  샘플: PO05, PO11, PO28, PO41, PO42 (AD), PO13, PO15, PO20, PO23 (CT)
        SMC027–SMC053 (16 samples)

Barcode 정보:
  h5ad obs['barcode'] or cell ID → cluster C18 필터링 가능
  sample_codes + leiden_codes로 per-sample per-cluster barcode 추출

Count matrix (이미 있음):
  tx_counts_by_donor_Excitatory_neuron.csv — donor level only (cluster별 아님)
```

---

## 검증 파이프라인 (단계별)

### Step 1: C18 cell barcode 추출 (즉시 가능, ~1시간)

```python
# h5py로 직접 barcode 추출
import h5py
import numpy as np

f = h5py.File(ADATA_PATH, 'r')

# barcode 컬럼 확인
obs_keys = list(f['obs'].keys())
# 보통 'barcode', '_index', 'obs_names' 중 하나

# C18 세포 마스크
leiden_codes = f['obs']['leiden']['codes'][:]
leiden_cats = f['obs']['leiden']['categories'][:].astype(str)
sample_codes = f['obs']['sample']['codes'][:]
sample_cats = f['obs']['sample']['categories'][:].astype(str)
cell_type_codes = f['obs']['cell_type']['codes'][:]
cell_types = f['obs']['cell_type']['categories'][:].astype(str)

exc_idx = np.where(cell_types == 'Excitatory neuron')[0][0]
c18_idx = np.where(leiden_cats == '18')[0][0]

c18_mask = (leiden_codes == c18_idx) & (cell_type_codes == exc_idx)

# 샘플별 C18 barcode 추출
for s_idx, samp in enumerate(sample_cats):
    mask = c18_mask & (sample_codes == s_idx)
    barcodes = f['obs']['_index'][mask].astype(str)  # 또는 'barcode'
    # → {samp}_C18_barcodes.txt 저장
```

**출력**: `reports/c18_barcodes/{sample_id}_C18_barcodes.txt` (25 파일)

---

### Step 2: BAM 파일 위치 확인 및 접근 (0.5일)

```bash
# BAM 파일 탐색
ls /home/dhkim1674/Project_AD_with_refTSS_novel/02_BAM/
# 또는
find /home/dhkim1674/ -name "*.bam" -path "*/Long_Read/*" | head -10

# CB tag 확인 (cell barcode 포함 여부)
samtools view -H PO05_sorted.bam | grep CB
samtools view PO05_sorted.bam | head -5 | cut -f12-20
```

---

### Step 3: C18 BAM subset 추출 (samtools, per sample, ~2시간/sample)

```bash
# picard FilterSamReads 또는 samtools view + 파이썬 필터
# 방법 A: samtools + grep
samtools view -h PO05.bam | \
  awk 'BEGIN{while(getline < "PO05_C18_barcodes.txt") bc[$1]=1}
       /^@/ || ($0 ~ /CB:Z:/ && substr($0, index($0,"CB:Z:")+5, 16) in bc)' | \
  samtools view -bS - > PO05_C18.bam

# 방법 B: Python pysam (정확도 높음)
import pysam

def filter_bam_by_barcodes(in_bam, barcodes_set, out_bam):
    with pysam.AlignmentFile(in_bam, 'rb') as bam_in, \
         pysam.AlignmentFile(out_bam, 'wb', header=bam_in.header) as bam_out:
        for read in bam_in:
            try:
                cb = read.get_tag('CB')
                if cb in barcodes_set:
                    bam_out.write(read)
            except KeyError:
                pass
```

**출력**: `{sample_id}_C18.bam` (25 파일, 추정 크기: 원본의 ~5%)

---

### Step 4: IsoQuant로 KIF21B isoform 재계산 (per cluster)

```bash
# 기존 IsoQuant GTF를 reference로 사용
isoquant.py \
  --reference /path/to/genome.fa \
  --genedb /path/to/existing_annotation.gtf \
  --bam PO05_C18.bam PO11_C18.bam PO28_C18.bam PO41_C18.bam PO42_C18.bam \
       PO13_C18.bam PO15_C18.bam PO20_C18.bam PO23_C18.bam \
  --data_type 10x \
  --output isoquant_C18_output/ \
  --count_exons

# 비교용: C10+C11 (canonical L4) 및 C19 (L5) 동일하게 수행
```

---

### Step 5: KIF21B isoform usage 비교 (핵심 분석)

```python
# KIF21B isoform usage per cluster per sample
clusters_to_compare = {
    'C18_L4*': 'isoquant_C18_output/',
    'C10C11_L4': 'isoquant_C10C11_output/',
    'C19_L5': 'isoquant_C19_output/',
}
target_isoforms = {
    'CT_motor': 'transcript293004.chr1.nic',  # CT-enriched, motor domain intact
    'AD_WD40': 'KIF21B-203',                  # AD-enriched, WD40 repeat
}

# Per cluster per sample: isoform usage fraction
# Compare: C18_AD vs C18_CT vs C10C11_AD vs C10C11_CT
```

---

## 예상 결과 해석 기준

| 결과 | 해석 |
|------|------|
| C18 CT motor fraction < C10+C11 CT | RBFOX1 저발현 → exon17 inclusion 감소 → C18 이미 취약 상태 |
| C18 AD motor fraction << C18 CT | AD에서 C18-specific switch 확인 → 핵심 발견 지지 |
| C18 AD vs C18 CT non-sig | switch가 bulk Exc 수준에서 일어나지만 C18-specific이 아님 |
| KIF21B switch 모든 cluster 균등 | Cell type에 무관한 switch → AD 전체 Exc 균등 취약성 |

---

## 예상 소요 자원

| 단계 | 시간 | 필요 도구 |
|------|------|-----------|
| Step 1 (barcode 추출) | 1시간 | Python + h5py (즉시 가능) |
| Step 2 (BAM 확인) | 0.5일 | samtools |
| Step 3 (BAM 필터링) | ~2시간/sample × 25 = ~2일 | pysam / samtools |
| Step 4 (IsoQuant) | ~6시간/cluster | IsoQuant v3 |
| Step 5 (분석) | 0.5일 | Python + pandas |
| **총합** | **~5일** | GPU 불필요 |

---

## 즉시 수행 가능한 선행 단계

```python
# Step 1 바로 실행 가능 — h5ad에서 barcode 추출
# 필요: h5ad obs에 barcode/cell ID 컬럼 존재 여부 확인
import h5py
f = h5py.File(ADATA_PATH, 'r')
print(list(f['obs'].keys()))  # 'barcode' 또는 '_index' 확인
f.close()
```

barcode 컬럼이 있으면 Step 1 → 즉시 착수 가능.  
BAM 접근 권한은 dhkim1674 서버에 별도 확인 필요.

---

## 대안: 기존 count matrix 활용 (빠른 버전, ~1일)

현재 `tx_counts_by_donor_Excitatory_neuron.csv`는 donor-level (cluster 구분 없음).  
만약 **cluster-specific count matrix**가 이미 생성되어 있다면 Step 3-4 생략 가능.

```bash
# 확인
ls /home/dhkim1674/Project_AD_with_refTSS_novel/04_Counts/Long_Read/AD/AD_vs_AC_vs_CT/counts_by_cluster/
# 또는
ls /home/dhkim1674/Project_AD_with_refTSS_novel/04_Counts/Long_Read/AD/AD_vs_AC_vs_CT/counts_by_donor/
```
