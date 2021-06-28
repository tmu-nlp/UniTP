#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename index_t, typename scalar_t>
__global__ void kernel_block_max(
  const torch::PackedTensorAccessor32<index_t,2,torch::RestrictPtrTraits> sections,
  const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> values,
  torch::PackedTensorAccessor32<bool,2,torch::RestrictPtrTraits> bool_max,
  torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> sec2max,
  torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> idx_max,
  torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> val_max) {
  const int bid = blockIdx.x;
  const int batch_len = sections.size(1);
  const int num_sections = idx_max.size(1);
  int idx;
  scalar_t val;

  for (int i = 0; i < batch_len; i++) {
    idx = sections[bid][i]; val = values[bid][i];
    if (idx_max[bid][idx] < 0 or val_max[bid][idx] < val) {
      idx_max[bid][idx] = i; val_max[bid][idx] = val;
    }
  }

  for (int i = 0; i < batch_len; i++) {
    sec2max[bid][i] = idx_max[bid][sections[bid][i]];
  }

  idx = 0;
  do { bool_max[bid][idx_max[bid][idx++]] = true; }
  while (idx < num_sections and idx_max[bid][idx] >= 0);
}

std::vector<torch::Tensor> blocky_max_cuda(
    torch::Tensor sections, // long
    torch::Tensor values) { // float
  const int batch_size = sections.size(0);
  const int batch_len = sections.size(1);
  const auto &index_type = sections.dtype();
  bool long_index = index_type == torch::kLong;
  auto svt = sections.max(); svt += 1;
  const int section_volume = long_index ? svt.item<long>() : svt.item<int>();
  // std::printf("section_volume = %d\n", section_volume);

  const auto &device = sections.device();
  const auto &int_options = torch::TensorOptions().dtype(torch::kInt).device(device);
  const auto &bool_options = torch::TensorOptions().dtype(torch::kBool).device(device);
  const auto &val_options = torch::TensorOptions().dtype(values.scalar_type()).device(device);

  auto bool_max = torch::zeros({batch_size, batch_len}, bool_options);
  auto idx_max = torch::zeros({batch_size, section_volume}, int_options); idx_max -= 1;
  // auto idx_max = torch::full({batch_size, section_volume}, int_options, -1);
  auto val_max = torch::empty({batch_size, section_volume}, val_options);
  auto sec2max = torch::empty({batch_size, batch_len}, int_options);

  if (long_index) 
    AT_DISPATCH_ALL_TYPES(values.scalar_type(), "kernel_block_max", ([&] {
      kernel_block_max<long, scalar_t><<<batch_size, 1>>>(
          sections.packed_accessor32<long,2,torch::RestrictPtrTraits>(),
          values.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
          bool_max.packed_accessor32<bool,2,torch::RestrictPtrTraits>(),
          sec2max.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
          idx_max.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
          val_max.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
    }));
  else
    AT_DISPATCH_ALL_TYPES(values.scalar_type(), "kernel_block_max", ([&] {
      kernel_block_max<int, scalar_t><<<batch_size, 1>>>(
          sections.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
          values.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
          bool_max.packed_accessor32<bool,2,torch::RestrictPtrTraits>(),
          sec2max.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
          idx_max.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
          val_max.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
    }));
  return {bool_max, sec2max};
}

template <typename index_t>
__global__ void kernel_fan_in(
  const torch::PackedTensorAccessor32<bool,3,torch::RestrictPtrTraits> bool_matrix,
  const torch::PackedTensorAccessor32<index_t,1,torch::RestrictPtrTraits> dis_length,
  torch::PackedTensorAccessor32<index_t,2,torch::RestrictPtrTraits> comp_id,
  torch::PackedTensorAccessor32<bool,2,torch::RestrictPtrTraits> comp_check) {
  // batch & child index
  const int bid = blockIdx.x;
  const int yid = threadIdx.x;
  // size & pad
  const int max_comp_batch = blockDim.x;
  const int max_comp_seq = dis_length[bid];
  int s_comp_id = max_comp_batch, xid;
  bool triggered = false, symmetric_lhs;

  if (yid < max_comp_seq){
    for (xid = 0; xid < max_comp_seq; xid++) {
      symmetric_lhs = bool_matrix[bid][yid][xid];
      if (symmetric_lhs != bool_matrix[bid][xid][yid]) {
        comp_check[bid][yid] = false;
        // std::printf("Asymmetric %d, %d\n", bid, yid);
        return;
      }
      if (symmetric_lhs) {
        if (xid < s_comp_id) s_comp_id = xid;
        triggered = s_comp_id < max_comp_batch;

        if (triggered and not bool_matrix[bid][s_comp_id][xid]) {
          // std::printf("Diff.0 %d, %d\n", bid, yid);
          comp_check[bid][yid] = false;
          return; // 
        }
      } else if (triggered and bool_matrix[bid][s_comp_id][xid]) {
        // std::printf("Diff.1 %d, %d\n", bid, yid);
        comp_check[bid][yid] = false;
        return;
      }
    }
    if (triggered) {
      xid = s_comp_id - 1;
      while (xid >= 0) {
        if (bool_matrix[bid][s_comp_id][xid]) {
          // std::printf("Diff.P %d, %d\n", bid, yid);
          comp_check[bid][yid] = false;
          return;
        }
        xid--;
      }
      comp_id[bid][yid] = s_comp_id;
    } else
      // std::printf("Not Found %d, %d\n", bid, yid);
      comp_check[bid][yid] = false;
  } else {
    comp_id[bid][yid] = -1;
  }
}

template <typename index_t>
__global__ void kernel_catch_invalid_each(
  const torch::PackedTensorAccessor32<bool,2,torch::RestrictPtrTraits> comp_check,
  const torch::PackedTensorAccessor32<index_t,1,torch::RestrictPtrTraits> dis_length,
  torch::PackedTensorAccessor32<index_t,2,torch::RestrictPtrTraits> comp_id,
  torch::PackedTensorAccessor32<bool,3,torch::RestrictPtrTraits> bool_matrix) {
  const int bid = blockIdx.x;
  const int yid = blockIdx.y;
  const int xid = blockIdx.z;
  const int length = dis_length[bid];
  if (xid >= length or yid >= length) return;
  const bool on_eye = xid == yid;
  const bool checked_x = comp_check[bid][xid];
  const bool checked_y = comp_check[bid][yid];

  if (checked_x and checked_y) return;
  else {
    if (on_eye) comp_id[bid][yid] = yid;
    if (on_eye and not checked_x and not checked_x) bool_matrix[bid][yid][xid] = on_eye;
    else {
      if (not checked_x) bool_matrix[bid][yid][xid] = on_eye;
      if (not checked_y) bool_matrix[bid][yid][xid] = on_eye;
    } // std::printf(" [%d, %d, %d] = %c\n", bid, yid, xid, on_eye ? 'T' : 'X');
  }
}

template <typename index_t>
__global__ void kernel_catch_invalid_all(
  const torch::PackedTensorAccessor32<bool,2,torch::RestrictPtrTraits> comp_check,
  const torch::PackedTensorAccessor32<index_t,1,torch::RestrictPtrTraits> dis_length,
  torch::PackedTensorAccessor32<index_t,2,torch::RestrictPtrTraits> comp_id,
  torch::PackedTensorAccessor32<bool,3,torch::RestrictPtrTraits> bool_matrix) {
  const int bid = blockIdx.x;
  const int yid = blockIdx.y;
  const int xid = blockIdx.z;
  const int length = dis_length[bid];
  if (xid >= length or yid >= length) return;
  const bool checked_x = comp_check[bid][xid];
  const bool checked_y = comp_check[bid][yid];

  if (checked_x and checked_y) return;
  else {
    comp_id[bid][yid] = length;
    bool_matrix[bid][yid][xid] = checked_x == checked_y;
  }
}

template <typename index_t>
__global__ void kernel_sort_comp_id(
  const torch::PackedTensorAccessor32<index_t,2,torch::RestrictPtrTraits> comp_id_x,
  torch::PackedTensorAccessor32<index_t,2,torch::RestrictPtrTraits> comp_id_y,
  torch::PackedTensorAccessor32<index_t,3,torch::RestrictPtrTraits> id_map) {
  // batch & child index
  const int bid = blockIdx.x;
  const int max_comp_per_batch = comp_id_x.size(1);
  int map_size = 0, ptr, offset, stride, level;
  index_t key, value, comp_id;

  // std::printf("bid, yid, dim.x, dim.y = %d %d %d %d\n", bid, yid, blockDim.x, blockDim.y);
  for (int yid = 0; yid < max_comp_per_batch; yid++) {
    comp_id = comp_id_x[bid][yid];
    if (comp_id < 0) {
      comp_id_y[bid][yid] = comp_id;
    } else {
      // start of std::map<index_t, index_t> replacement
      if (map_size) {
        value = -1; offset = 0; level = 1;
        while (true) {
          stride = map_size >> level;
          ptr = offset + stride;
          if (ptr >= map_size) break;
          key = id_map[bid][ptr][0];
          if (key == comp_id) {
            value = id_map[bid][ptr][1];
            break;
          } else {
            if (key < comp_id) offset = ptr + 1;
            level++;
            if (stride < 1) break;
          }
        }
        if (value < 0) {
          id_map[bid][map_size][0] = comp_id;
          id_map[bid][map_size][1] = map_size;
          value = map_size;
          map_size++;
        }
      } else {
        id_map[bid][map_size][0] = comp_id;
        id_map[bid][map_size][1] = map_size;
        value = map_size;
        map_size++;
      }
      // end of std::map<index_t, index_t> replacement
      comp_id_y[bid][yid] = value;
    }
  }
  while (map_size < max_comp_per_batch) {
    id_map[bid][map_size][0] = -1;
    id_map[bid][map_size][1] = -1;
    map_size++;
  }
}

template <typename index_t>
__global__ void kernel_sort_components(
  const torch::PackedTensorAccessor32<index_t,3,torch::RestrictPtrTraits> id_map,
  const torch::PackedTensorAccessor32<bool,3,torch::RestrictPtrTraits> bool_matrix,
  torch::PackedTensorAccessor32<bool,3,torch::RestrictPtrTraits> components) {
  const int bid = blockIdx.x;
  const int yid = blockIdx.y;
  const int xid = blockIdx.z;
  
  const index_t src = id_map[bid][yid][0];
  const index_t dst = id_map[bid][yid][1];

  if (src < 0 or dst < 0) return;

  components[bid][dst][xid] = bool_matrix[bid][src][xid];
}

template <typename index_t>
__global__ void kernel_sparse_setup(
  const torch::PackedTensorAccessor32<index_t,1,torch::RestrictPtrTraits> dis_length,// [dis_bs]
  const torch::PackedTensorAccessor32<index_t,1,torch::RestrictPtrTraits> dis_bid,   // [dis_bs]
  const torch::PackedTensorAccessor32<index_t,2,torch::RestrictPtrTraits> dis_sid,   // [dis_bs, dis_sl]
  const torch::PackedTensorAccessor32<bool,2,torch::RestrictPtrTraits> dis_max,      // [dis_bs, dis_sl]
  torch::PackedTensorAccessor32<bool,2,torch::RestrictPtrTraits> con_fence) {        // [    bs,     sl]
  const int bid = blockIdx.x; // dis_bs
  const int sid = blockIdx.y; // dis_sl
  // const bool is_max = ;
  if (sid >= dis_length[bid]) return;

  con_fence[dis_bid[bid]][dis_sid[bid][sid]] = dis_max[bid][sid]; // 64bit access?
}

template <typename index_t>
__global__ void kernel_connect_discontinuity(
  const torch::PackedTensorAccessor32<index_t,1,torch::RestrictPtrTraits> dis_length,
  const torch::PackedTensorAccessor32<index_t,1,torch::RestrictPtrTraits> dis_bid,
  const torch::PackedTensorAccessor32<index_t,2,torch::RestrictPtrTraits> dis_sid,
  const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> sec2max,
  const torch::PackedTensorAccessor32<bool,2,torch::RestrictPtrTraits> dis_max,
  torch::PackedTensorAccessor32<index_t,2,torch::RestrictPtrTraits> sections) {
  const int d_bid = blockIdx.x; // bs
  const int d_sid = blockIdx.y; // sl
  if (dis_max[d_bid][d_sid] or d_sid >= dis_length[d_bid]) return; // skip head to stablize it

  const int c_bid = dis_bid[d_bid];
  const int c_sid = dis_sid[d_bid][d_sid];
  const int c_hid = dis_sid[d_bid][sec2max[d_bid][d_sid]];
  sections[c_bid][c_sid] = sections[c_bid][c_hid]; // stablize heads
}

std::vector<torch::Tensor> predict_disco_sections_cuda(
    torch::Tensor dis_matrix, // float
    const float threshold,
    const bool catch_any_invalid,
    torch::Tensor dis_length, // long
    torch::Tensor dis_bid,    // long
    torch::Tensor dis_sid,    // long
    torch::Tensor con_fence,  // bool
    torch::Tensor dis_score) {// float

  const int batch_size = dis_matrix.size(0); // each batch unit is independent without shared SM memory
  const int max_comp_per_batch = dis_matrix.size(1); // max(any fan_out) < 512
  const int batch_len = dis_sid.size(1);
  const dim3 batch_square_dim3(batch_size, max_comp_per_batch, max_comp_per_batch);
  const dim3 batch_linear_dim3(batch_size, max_comp_per_batch);
  const auto &device = dis_length.device();
  const auto &int_options = torch::TensorOptions().dtype(dis_length.dtype()).device(device);
  const auto &bool_options = torch::TensorOptions().dtype(con_fence.dtype()).device(device);

  auto bool_matrix_x = dis_matrix > threshold;
  auto bool_matrix_y = torch::transpose(bool_matrix_x, 1, 2);

  auto comp_id_x = torch::zeros({batch_size, max_comp_per_batch}, int_options);
  auto comp_id_y = torch::zeros({batch_size, max_comp_per_batch}, int_options);
  auto comp_id_map = torch::zeros({batch_size, max_comp_per_batch, 2}, int_options);
  auto comp_check = torch::ones({batch_size, max_comp_per_batch}, bool_options);
  auto components = torch::zeros({batch_size, max_comp_per_batch, max_comp_per_batch}, bool_options);

  AT_DISPATCH_INDEX_TYPES(dis_length.scalar_type(), "kernel_fan_in", ([&] {
    kernel_fan_in<index_t><<<batch_size, max_comp_per_batch>>>(
        bool_matrix_x.packed_accessor32<bool,3,torch::RestrictPtrTraits>(),
        dis_length.packed_accessor32<index_t,1,torch::RestrictPtrTraits>(),
        comp_id_x.packed_accessor32<index_t,2,torch::RestrictPtrTraits>(),
        comp_check.packed_accessor32<bool,2,torch::RestrictPtrTraits>());
  }));

  // AT_DISPATCH_INDEX_TYPES(dis_length.scalar_type(), "kernel_fan_in", ([&] {
  //   kernel_fan_in<index_t><<<batch_size, max_comp_per_batch>>>(
  //       bool_matrix_y.packed_accessor32<bool,3,torch::RestrictPtrTraits>(),
  //       dis_length.packed_accessor32<index_t,1,torch::RestrictPtrTraits>(),
  //       comp_id_y.packed_accessor32<index_t,2,torch::RestrictPtrTraits>(),
  //       comp_check.packed_accessor32<bool,2,torch::RestrictPtrTraits>());
  // }));

  // const auto &invalid = torch::where(comp_check);
  // std::tuple<torch::Tensor, torch::Tensor> thresholds_obj = (dis_matrix - threshold).sort();
  // const torch::Tensor &thresholds = std::get<1>(thresholds_obj);

  // comp_check &= comp_id_x == comp_id_y;

  if (catch_any_invalid)
    AT_DISPATCH_INDEX_TYPES(dis_length.scalar_type(), "kernel_catch_invalid_each", ([&] {
      kernel_catch_invalid_each<index_t><<<batch_square_dim3, 1>>>(
          comp_check.packed_accessor32<bool,2,torch::RestrictPtrTraits>(),
          dis_length.packed_accessor32<index_t,1,torch::RestrictPtrTraits>(),
          comp_id_x.packed_accessor32<index_t,2,torch::RestrictPtrTraits>(),
          bool_matrix_x.packed_accessor32<bool,3,torch::RestrictPtrTraits>());
    }));
  else
    AT_DISPATCH_INDEX_TYPES(dis_length.scalar_type(), "kernel_catch_invalid_all", ([&] {
      kernel_catch_invalid_all<index_t><<<batch_square_dim3, 1>>>(
          comp_check.packed_accessor32<bool,2,torch::RestrictPtrTraits>(),
          dis_length.packed_accessor32<index_t,1,torch::RestrictPtrTraits>(),
          comp_id_x.packed_accessor32<index_t,2,torch::RestrictPtrTraits>(),
          bool_matrix_x.packed_accessor32<bool,3,torch::RestrictPtrTraits>());
    }));

  AT_DISPATCH_INDEX_TYPES(dis_length.scalar_type(), "kernel_sort_comp_id", ([&] {
    kernel_sort_comp_id<index_t><<<batch_size, 1>>>(
        comp_id_x.packed_accessor32<index_t,2,torch::RestrictPtrTraits>(),
        comp_id_y.packed_accessor32<index_t,2,torch::RestrictPtrTraits>(),
        comp_id_map.packed_accessor32<index_t,3,torch::RestrictPtrTraits>());
  }));

  AT_DISPATCH_INDEX_TYPES(dis_length.scalar_type(), "kernel_sort_components", ([&] {
    kernel_sort_components<index_t><<<batch_square_dim3, 1>>>(
        comp_id_map.packed_accessor32<index_t,3,torch::RestrictPtrTraits>(),
        bool_matrix_x.packed_accessor32<bool,3,torch::RestrictPtrTraits>(),
        components.packed_accessor32<bool,3,torch::RestrictPtrTraits>());
  }));

  const auto &blocky_max_c2h = blocky_max_cuda(comp_id_y + 1, dis_score);
  const auto &blocky_max = blocky_max_c2h[0];
  const auto &blocky_c2h = blocky_max_c2h[1];
  // const auto &num_comp = (comp_id_map.index({torch::indexing::Ellipsis, 0}) >= 0).sum(1);
  // const auto &max_comp = num_comp.max();
  // TODO select head from comp_id & dis_indice(dd)
  // use dd to swipe con_fence
  // cumu & reset comp in cumu

  AT_DISPATCH_INDEX_TYPES(dis_length.scalar_type(), "kernel_sparse_setup", ([&] {
    kernel_sparse_setup<index_t><<<batch_linear_dim3, 1>>>(
        dis_length.packed_accessor32<index_t,1,torch::RestrictPtrTraits>(),
        dis_bid.packed_accessor32<index_t,1,torch::RestrictPtrTraits>(),
        dis_sid.packed_accessor32<index_t,2,torch::RestrictPtrTraits>(),
        blocky_max.packed_accessor32<bool,2,torch::RestrictPtrTraits>(),
        con_fence.packed_accessor32<bool,2,torch::RestrictPtrTraits>());
  }));

  auto sections = con_fence.cumsum(1);

  AT_DISPATCH_INDEX_TYPES(dis_length.scalar_type(), "kernel_connect_discontinuity", ([&] {
    kernel_connect_discontinuity<index_t><<<batch_linear_dim3, 1>>>(
        dis_length.packed_accessor32<index_t,1,torch::RestrictPtrTraits>(),
        dis_bid.packed_accessor32<index_t,1,torch::RestrictPtrTraits>(),
        dis_sid.packed_accessor32<index_t,2,torch::RestrictPtrTraits>(),
        blocky_c2h.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
        blocky_max.packed_accessor32<bool,2,torch::RestrictPtrTraits>(),
        sections.packed_accessor32<index_t,2,torch::RestrictPtrTraits>());
  }));

  return {sections, comp_id_y, components, comp_check};
}