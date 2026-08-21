# lufloat

[![Rust](https://img.shields.io/badge/Rust-2024-orange?logo=rust)](https://rust-lang.org/)

[![Linux](https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black)](https://www.linux.org/)
[![Rust](https://img.shields.io/badge/Rust-2024-orange?logo=rust)](https://www.rust-lang.org/)
[![ROCm/HIP](https://img.shields.io/badge/ROCm-HIP-blue?logo=amd)](https://rocm.docs.amd.com)
[![License](https://img.shields.io/badge/License-AGPL--3.0-blue)](LICENSE)

## Fastest FP16 Math and AI Library for AMD APUs

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

* AMD APU
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