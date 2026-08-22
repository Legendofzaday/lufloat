# lufloat

![ROCm/HIP](https://img.shields.io/badge/ROCm%2FHIP-black?logo=amd&logoColor=white&logoSize=auto)
![Rust](https://img.shields.io/badge/Rust-black?logo=rust&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-black?logo=linux&logoColor=white)
![AGPL v3.0](https://img.shields.io/badge/AGPL--v3.0-black?logo=gnu&logoColor=white)

## Fastest FP16 Math and AI Library for AMD APUs

**⚠️ WORK IN PROGRESS ALPHA**

**Note:** Tested with AI Max+ 395 on Fedora.

| Operation | Size | Data Read | Data Write | Time | Bandwidth | Utilization |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `negative_mask` | `2^34` | 32GiB | 32GiB |  288ms | 238.6GB/s | 93.2% |
| `negative_mask_inplace` | `2^35` | 64GiB | 64GiB |  576ms | 238.6GB/s | 93.2% |
| `positive_mask` | `2^34` | 32GiB | 32GiB |  288ms | 238.6GB/s | 93.2% |
| `positive_mask_inplace` | `2^35` | 64GiB | 64GiB |  576ms | 238.6GB/s | 93.2% |

## Usage

```bash
cargo add lufloat
```

```rust
use lufloat::{Arena, UnifiedBuffer};

let arena = Arena::new(2048);
let mut buffer = UnifiedBuffer::new(&arena, 2048);
let input_data = buffer.slice_mut();

input_data[0] = 0b1_01111_0000000000; // -1.0
input_data[1] = 0b0_00000_0000000000; // 0.0
input_data[2] = 0b0_01111_0000000000; // 1.0

buffer.positive_mask_inplace();
let output_data = buffer.slice();

println!("The first 3 elements are: {:?}", &output_data[..3]); // 0.0, 1.0, 1.0
```

## Requirements

* AMD RDNA 3 APU
* Linux
* ROCm/HIP `hipcc`
* binutils `ar`

## License

lufloat
Copyright (C) 2026  Thopuri Omkar Eeswar

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>