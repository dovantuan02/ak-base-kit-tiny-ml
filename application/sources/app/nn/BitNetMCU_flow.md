# BitNetMCU Inference Pipeline — CH32V003 (RISC-V, no HW MUL)

## 1. Tổng quan luồng dữ liệu

```
MNIST 28×28 → resize → 16×16 int8 → BitMnistInference() → prediction (0..9)
```

```mermaid
flowchart TD
    subgraph Input
        A1["Raw image 28×28"] --> A2["Resize 16×16<br/>Scale to int8 (-128..127)"]
        A2 --> A3["input_data[256] int8"]
    end

    subgraph BitMnistInference
        A3 --> B["Copy → int32 buffer<br/>layer_out[256]"]
        B --> C["Conv block ×64 channels<br/>(depthwise separable)"]
        C --> D["Fuse channels → layer_in[256] int32"]
        D --> E["ReLUNorm → int8[256]"]
        E --> F["FC L11: 256→96<br/>2-bit ternary"]
        F --> G["ReLUNorm → int8[96]"]
        G --> H["FC L13: 96→64<br/>4-bit symmetric"]
        H --> I["ReLUNorm → int8[64]"]
        I --> J["FC L15: 64→10<br/>4-bit symmetric"]
        J --> K["ReLUNorm → int8[10]<br/>Return argmax"]
    end

    K --> L["Prediction (0..9)"]

    style A1 fill:#f9f,stroke:#333
    style K fill:#bbf,stroke:#333
    style L fill:#bfb,stroke:#333
```

---

## 2. Chi tiết Conv block (depthwise separable)

```mermaid
flowchart TD
    subgraph Per-Channel Loop
        direction LR
        IN["input_raw[256] int32<br/>(copy per channel)"] --> L2["L2 Conv2d 3×3<br/>1→64 ch, shift=4<br/>16² → 14²"]
        L2 --> L4["L4 Conv2d 3×3<br/>64→64 ch, groups=64<br/>14² → 12²"]
        L4 --> L6["L6 MaxPool2d 2×2<br/>12² → 6²"]
        L6 --> L7["L7 Conv2d 3×3<br/>64→64 ch, groups=64<br/>6² → 4²"]
        L7 --> L9["L9 MaxPool2d 2×2<br/>4² → 2²"]
        L9 --> OUT["outputptr[4] int32<br/>→ layer_in @ offset ch×4"]
    end

    subgraph Channel Combine
        OUT --> COMBINE["layer_in[64ch × 4val = 256] int32"]
    end

    IN -->|"Re-init from input<br/>each channel"| L2
```

**Depthwise separable** = mỗi channel conv độc lập (groups=64). 64 ch × 64 ch × 3×3 → 64 conv riêng, ko cross-channel.

---

## 3. processconv33ReLU — Conv2d 3×3 fused với ReLU

```mermaid
flowchart TD
    A["activations[xy_input × xy_input] int32"] --> B["weights[9] int8<br/>(SRAM copy)"]
    B --> C["Slide kernel 3×3<br/>stride=1, padding=0"]
    C --> D["Unrolled MAC (9 mul):<br/>sum = Σ w[i] × in[i]"]
    D --> E{"sum < 0?"}
    E -->|"Yes"| F["output = 0 (ReLU)"]
    E -->|"No"| G["sum >> n_shift<br/>(n_shift=4)"]
    G --> H["output = sum"]
    F --> I["output[] → next layer"]
    H --> I
```

**Toán tử kernel (unrolled, ko loop con):**

```
sum  = w[0]*in[0]  + w[1]*in[1]  + w[2]*in[2]
     + w[3]*in[0+] + w[4]*in[1+] + w[5]*in[2+]
     + w[6]*in[0+] + w[7]*in[1+] + w[8]*in[2+]
                            ↑ xy_input stride
```

**Shift thay vì nhân:** `sum >> 4` giảm scale từ 8-bit weight × 8-bit activation → int32, xuống lại int32 phù hợp cho layer sau.

---

## 4. processmaxpool22 — MaxPool 2×2

```mermaid
flowchart TD
    A["activations[xy_input × xy_input]"] --> B["Chia grid 2×2<br/>non-overlapping"]
    B --> C["Tìm max trong 2×2 patch:<br/>max(v[0], v[1],<br/>v[xy_input], v[xy_input+1])"]
    C --> D["output[xy_output × xy_output]<br/>xy_output = xy_input / 2"]
    D --> E["Trả về pointer cuối output"]
```

**In-place OK** — output buffer có thể trùng input buffer.

---

## 5. processfclayer — Fully Connected (multi-bit)

```mermaid
flowchart TD
    A["activations[int8 × n_input]"] --> B["Weight decoder"]
    B --> C{"bits_per_weight"}
    
    C -->|"1"| D1["32 weights/word<br/>bit 31 = sign (+/−)"]
    C -->|"2"| D2["16 weights/word<br/>bit 31 = sign<br/>bit 30 = ×2"]
    C -->|"4"| D3["8 weights/word<br/>bit 31 = sign<br/>bit 28-30 = scale ×1/2/4/8"]
    C -->|"64 (ternary)"| D4["10 trits/16 bit<br/>bit 17=0 → nonzero<br/>bit 16 = sign"]
    C -->|"8+4 / 8+8"| D5["Twos-complement<br/>4 or 8 bit signed int"]

    D1 --> E["sum += sign ? -in : in"]
    D2 --> E
    D3 --> E["sum += tmpsum + (tmpsum<<1) + (tmpsum<<2) + (tmpsum<<3)"]
    D4 --> E["sum += sign ? -in : in"]
    D5 --> E["sum += in × weight"]
    
    E --> F["output[i] = sum<br/>(int32)"]
    F --> G["i++ < n_output?"]
    G -->|"Yes"| A
    G -->|"No"| H["Done"]
```

### Chi tiết decoder cho từng loại weight:

| bpw | Tên | weights/word | Mỗi weight chiếm | Giá trị |
|---|---|---|---|---|
| 1 | Binary | 32 | 1 bit | 0 → +1, 1 → -1 |
| 2 | 2-bit symmetric | 16 | 2 bit | 00→0, 01→+1, 10→+2, 11→-1 |
| 4 | 4-bit symmetric (CH32V003) | 8 | 4 bit | sign + shift 0..3 → ±1/2/4/8 |
| 4 | 4-bit symmetric (generic) | 8 | 4 bit | sign + scale nibble → ±(1..15) |
| 64 | Ternary (10 trits) | — | 16 bit/10 weights | base3 encoding |
| 8+4 | 4-bit twos-complement | 8 | 4 bit | -8..7 |
| 8+8 | 8-bit twos-complement | 4 | 8 bit | -128..127 |

---

## 6. ReLUNorm — Normalization + ReLU

```mermaid
flowchart TD
    A["input[int32 × n_input]"] --> B["Find max_val + argmax"]
    B --> C["scale = max_val >> 7"]
    C --> D["shift = floor(log2(scale))"]
    D --> E["rounding = 1 << (shift-1)"]
    E --> F["For each i:"]
    F --> G{"input[i] < 0?"}
    G -->|"Yes"| H["output[i] = 0 (ReLU)"]
    G -->|"No"| I["tmp = (input[i] + rounding) >> shift"]
    I --> J{"tmp > 127?"}
    J -->|"Yes"| K["output[i] = 127 (clip)"]
    J -->|"No"| L["output[i] = tmp"]
    H --> M["i++"]
    K --> M
    L --> M
    M -->|"i < n_input"| F
    M -->|"Done"| N["return argmax"]
```

**Động:** shift phụ thuộc vào activation lớn nhất → scale range về [-127..127].

---

## 7. Ma trận layer đầy đủ (model test)

```mermaid
flowchart LR
    subgraph Input
        I["16×16 int8<br/>(256)"]
    end

    subgraph Conv
        L2["L2 Conv<br/>16²→14²<br/>8-bit weight"]
        L4["L4 Conv<br/>14²→12²<br/>8-bit weight"]
        L6["L6 Pool<br/>12²→6²"]
        L7["L7 Conv<br/>6²→4²<br/>8-bit weight"]
        L9["L9 Pool<br/>4²→2²"]
    end

    subgraph ConvOut
        CO["64ch × 4val<br/>= 256 int32"]
    end

    subgraph FC
        L11["L11 FC<br/>256→96<br/>2-bit ternary"]
        L13["L13 FC<br/>96→64<br/>4-bit sym"]
        L15["L15 FC<br/>64→10<br/>4-bit sym"]
    end

    subgraph Out
        O["argmax<br/>0..9"]
    end

    I -->|"×64 depthwise"| L2
    L2 --> L4
    L4 --> L6
    L6 --> L7
    L7 --> L9
    L9 --> CO
    CO -->|"ReLUNorm"| L11
    L11 -->|"ReLUNorm"| L13
    L13 -->|"ReLUNorm"| L15
    L15 -->|"ReLUNorm"| O

    style I fill:#e1d5e7
    style Conv fill:#dae8fc
    style FC fill:#d5e8d4
    style O fill:#ffcccc
```

### Tensor shape qua từng layer:

| Stage | Layer | Tensor shape | Phần tử | Loại |
|---|---|---|---|---|
| Input | — | 16×16 | 256 | int8 |
| L2 | Conv 3×3 | 14×14 | 196 | int32 |
| L4 | Conv 3×3 | 12×12 | 144 | int32 |
| L6 | MaxPool 2×2 | 6×6 | 36 | int32 |
| L7 | Conv 3×3 | 4×4 | 16 | int32 |
| L9 | MaxPool 2×2 | 2×2 | 4 | int32 |
| — | Stack 64ch | 64×2×2 | 256 | int32 |
| Norm | ReLUNorm | 256 | 256 | int8 |
| L11 | FC 256→96 | 96 | 96 | int32→int8 |
| L13 | FC 96→64 | 64 | 64 | int32→int8 |
| L15 | FC 64→10 | 10 | 10 | int32→int8 |

---

## 8. Debug logging system

```mermaid
flowchart TD
    subgraph DEBUG_LEVEL
        L0["0: OFF"]
        L1["1: STATS<br/>min/max/zero/neg/pos"]
        L2["2: HEAD<br/>+ first N values"]
        L3["3: FULL<br/>+ dump all values"]
    end

    subgraph Helpers
        H1["DBG_tensor_stats()<br/>label, cnt, min, max, z, n, p"]
        H2["DBG_tensor_head()<br/>label, first N int32"]
        H3["DBG_tensor_full()<br/>label, all int32, wrap cols"]
        H4["DBG_tensor_full_i8()<br/>label, all int8, wrap cols"]
    end

    L1 --> H1
    L2 --> H1 & H2
    L3 --> H1 & H2 & H3 & H4
```

**Output mẫu (DBG=1):**
```
[DBG] input_raw       cnt=256  min=-22   max=124   zero=0   neg=0   pos=0
[DBG] L2_conv         cnt=196  min=0     max=127   zero=32  neg=0   pos=164
[DBG] L4_conv         cnt=144  min=0     max=127   zero=28  neg=0   pos=116
[DBG] L6_pool         cnt=36   min=0     max=127   zero=4   neg=0   pos=32
[DBG] L7_conv         cnt=16   min=0     max=127   zero=2   neg=0   pos=14
[DBG] L9_pool         cnt=4    min=12    max=103   zero=0   neg=0   pos=4
[DBG] FC_in_after_norm cnt=256 min=0     max=127   zero=5   neg=0   pos=251
[DBG] L15_fc_out      cnt=10   min=-1    max=7     zero=0   neg=2   pos=8
[DBG] L15_norm_output cnt=10   min=0     max=127   zero=2   neg=0   pos=8
```

---

## 9. Tối ưu cho CH32V003 (RISC-V, ko MUL)

```
┌──────────────────────────────────────────────┐
│  CH32V003: RV32EC, 48 MHz, 2KB RAM, 16KB FLASH │
├──────────────────────────────────────────────┤
│  • processfclayer dùng shift thay multiply    │
│    sum += tmpsum<<1 (×2) thay vì sum += in*2 │
│  • Conv unrolled 3×3 — ko loop kernel         │
│  • MaxPool in-place — ko alloc thêm           │
│  • Weight packed bit — giảm flash              │
│  • int32 activation buffer — tránh overflow    │
│  • SRAM function attribute (comment out)       │
└──────────────────────────────────────────────┘
```

**processfclayer 4-bit (CH32V003 path):**
```c
// Không dùng multiply instruction
int32_t tmpsum = (weightChunk & 0x80000000) ? -in : in; // sign
sum += tmpsum;                                  // ×1
if (weightChunk & 0x10000000) sum += tmpsum<<1; // ×2
if (weightChunk & 0x20000000) sum += tmpsum<<2; // ×4
if (weightChunk & 0x40000000) sum += tmpsum<<3; // ×8
```

vs **generic path** (có MUL):
```c
int32_t tmpsum = (weightChunk & 0x80000000) ? -in : in;
sum += tmpsum;
sum += tmpsum * ((weightChunk>>(32-4-1))&0x0e); // dùng MUL
```
