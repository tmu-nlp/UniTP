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

  for (int i = 0; i < num_sections; i++) {
    idx = idx_max[bid][i];
    if (idx >= 0)
      bool_max[bid][idx] = true;
  }
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
  auto idx_max = torch::full({batch_size, section_volume}, -1, int_options);
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
        comp_check[bid][yid] = false; // std::printf("Asymmetric %d, %d\n", bid, yid);
        return;
      }
      if (symmetric_lhs) {
        if (xid < s_comp_id) s_comp_id = xid;
        triggered = s_comp_id < max_comp_batch;

        if (triggered and not bool_matrix[bid][s_comp_id][xid]) {
          comp_check[bid][yid] = false;
          return; // 
        }
      } else if (triggered and bool_matrix[bid][s_comp_id][xid]) {
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

template <typename index_t, typename scalar_t>
__global__ void kernel_write_2d_pad(
  const torch::PackedTensorAccessor32<index_t,1,torch::RestrictPtrTraits> dis_length,
  torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> hard_matrix) {
  const int bid = blockIdx.x;
  const int yid = blockIdx.y;
  const int xid = blockIdx.z;
  const int length = dis_length[bid];
  if (xid < length and yid < length) return;
  hard_matrix[bid][yid][xid] = 127.0; // upper bound is not so okay..
}

template <typename index_t, typename scalar_t>
__global__ void kernel_write_choice(
  torch::PackedTensorAccessor32<bool,2,torch::RestrictPtrTraits> hard_check,
  const torch::PackedTensorAccessor32<index_t,1,torch::RestrictPtrTraits> batch_idx,
  const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> hard_threshold,
  torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> threshold,
  torch::PackedTensorAccessor32<index_t,1,torch::RestrictPtrTraits> trials) {
  const int n_instances = hard_check.size(1);
  const int bid = blockIdx.x;
  const int jid = batch_idx[bid];
  bool found = false;
  for(int i = 0; i < n_instances; i++)
    if (hard_check[bid][i])
      if (found)
        hard_check[bid][i] = false;
      else {
        found = true;
        trials[jid] += i + 1;
        threshold[jid] = hard_threshold[bid][i];
      }
  if (not found) trials[jid] += n_instances;
}

template <typename index_t>
__global__ void kernel_write_back(
  const torch::PackedTensorAccessor32<index_t,2,torch::RestrictPtrTraits> hard_choice,
  const torch::PackedTensorAccessor32<bool,4,torch::RestrictPtrTraits> hard_bool_matrices,
  const torch::PackedTensorAccessor32<index_t,3,torch::RestrictPtrTraits> hard_comp_id,
  const torch::PackedTensorAccessor32<index_t,1,torch::RestrictPtrTraits> hard_length,
  const torch::PackedTensorAccessor32<index_t,1,torch::RestrictPtrTraits> batch_idx,
  torch::PackedTensorAccessor32<bool,3,torch::RestrictPtrTraits> bool_matrix,
  torch::PackedTensorAccessor32<index_t,2,torch::RestrictPtrTraits> comp_id,
  torch::PackedTensorAccessor32<bool,2,torch::RestrictPtrTraits> comp_check) {
  const int cid = blockIdx.x;
  const int yid = blockIdx.y;
  const int bid = hard_choice[cid][0]; // 0
  const int length = hard_length[bid];
  // std::printf("K.bid=%d .yid=%d len=%d\n", bid, yid, length);
  if (yid >= length) return;
  // std::printf("In:%d %d\n", cid, yid);
  const int iid = hard_choice[cid][1]; // 1
  // std::printf("  %d\n", iid);
  const int jid = batch_idx[bid]; // 1
  // std::printf(" %d:%d\n", iid, jid);
  comp_id[jid][yid] = hard_comp_id[bid][iid][yid];
  comp_check[jid][yid] = true;
  for(int xid = 0; xid < length; xid ++)
    bool_matrix[jid][yid][xid] = hard_bool_matrices[bid][iid][yid][xid];
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

template <typename index_t, typename scalar_t>
__global__ void kernel_write_errors(
  const torch::PackedTensorAccessor32<bool,1,torch::RestrictPtrTraits> check,
  torch::PackedTensorAccessor32<index_t,1,torch::RestrictPtrTraits> trials,
  torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> threshold) {
  const int bid = blockIdx.x;
  if (not check[bid]) {
    threshold[bid] = -1.0;
    trials[bid] += 1; }
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
  const int dis_len = dis_length[bid];
  // const bool is_max = ;
  if (sid >= dis_len) return;
  const int d_bid = dis_bid[bid];
  const int d_sid = dis_sid[bid][sid];
  const int sid_ = sid + 1;
  const int d_sid_ = d_sid + 1;

  con_fence[d_bid][d_sid] = dis_max[bid][sid]; // 64bit access?

  if (d_sid_ < con_fence.size(1) and sid_ < dis_len and d_sid_ < dis_sid[bid][sid_] and not con_fence[d_bid][d_sid_])
    con_fence[d_bid][d_sid_] = true;
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
    torch::Tensor dis_length, // long
    torch::Tensor dis_bid,    // long
    torch::Tensor dis_sid,    // long
    torch::Tensor con_fence,  // bool
    torch::Tensor dis_score,  // float
    const float threshold,
    const bool group_all_invalid,
    int n_parallel) {

  const int batch_size = dis_matrix.size(0); // each batch unit is independent without shared SM memory
  const int max_comp_per_batch = dis_matrix.size(1); // max(any fan_out) < 512
  const int batch_len = dis_sid.size(1);
  const dim3 batch_square_dim3(batch_size, max_comp_per_batch, max_comp_per_batch);
  const dim3 batch_linear_dim3(batch_size, max_comp_per_batch);
  const auto &device = dis_length.device();
  const auto &int_options = torch::TensorOptions().dtype(dis_length.dtype()).device(device);
  const auto &bool_options = torch::TensorOptions().dtype(con_fence.dtype()).device(device);
  const auto &scalar_options = torch::TensorOptions().dtype(dis_matrix.dtype()).device(device);

  auto bool_matrix_x = dis_matrix > threshold;
  auto bool_matrix_y = torch::transpose(bool_matrix_x, 1, 2);

  auto comp_id_x = torch::zeros({batch_size, max_comp_per_batch}, int_options);
  auto comp_id_y = torch::zeros({batch_size, max_comp_per_batch}, int_options);
  auto comp_id_map = torch::zeros({batch_size, max_comp_per_batch, 2}, int_options);
  auto comp_check = torch::ones({batch_size, max_comp_per_batch}, bool_options);
  auto components = torch::zeros({batch_size, max_comp_per_batch, max_comp_per_batch}, bool_options);
  auto final_thresholds = torch::full({batch_size}, threshold, scalar_options);
  auto trials = torch::ones({batch_size}, int_options);

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
  // comp_check &= comp_id_x == comp_id_y;

  auto hard_batch = comp_check.all(1).logical_not();
  if (hard_batch.any().item<bool>()) {
    const bool long_index = dis_length.dtype() == torch::kLong;

    if (n_parallel != 0) {
      auto hard_length = dis_length.index({hard_batch}).contiguous(); // copy if necessary
      auto hlm = hard_length.max();
      int max_comp_per_hard_batch = long_index ? hlm.item<long>() : hlm.item<int>();

      auto hard_slice = torch::indexing::Slice(0, max_comp_per_hard_batch);
      auto hard_matrix = dis_matrix.index({hard_batch, hard_slice, hard_slice}).contiguous();
      
      auto hbs = hard_batch.sum();
      int hard_batch_size = long_index ? hbs.item<long>() : hbs.item<int>();
      auto hard_batch_square_dim3 = dim3(hard_batch_size, max_comp_per_hard_batch, max_comp_per_hard_batch);

      auto batch_idx = torch::arange(batch_size, int_options).index({hard_batch}).contiguous();

      if (n_parallel < 0) {n_parallel = 512 / hard_batch_size;}
      int sq = max_comp_per_hard_batch * max_comp_per_hard_batch;
      int wt = sq / n_parallel;
      if (sq % n_parallel) wt ++;

      if (long_index)
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(hard_matrix.scalar_type(), "kernel_write_2d_pad", ([&] {
          kernel_write_2d_pad<long, scalar_t><<<hard_batch_square_dim3, 1>>>(
            hard_length.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
            hard_matrix.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
        }));
      else
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(hard_matrix.scalar_type(), "kernel_write_2d_pad", ([&] {
          kernel_write_2d_pad<int, scalar_t><<<hard_batch_square_dim3, 1>>>(
            hard_length.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
            hard_matrix.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
        }));

      auto seq_threshold = hard_matrix.reshape({hard_batch_size, -1}); // shorter dis_length seq_threshold.index_put_({seq_threshold <= 0}, 1); 
      auto seq_distances = seq_threshold - threshold; seq_distances *= seq_distances;
      std::tuple<torch::Tensor, torch::Tensor> thresholds_obj = seq_distances.sort();
      auto seq_indices = torch::arange(hard_batch_size, int_options).unsqueeze(-1);
      seq_threshold = seq_threshold.index({seq_indices, std::get<1>(thresholds_obj)});

      for (int i = 0; i < wt; i++) {
        auto thres = seq_threshold.index({torch::indexing::Slice(), torch::indexing::Slice(i*n_parallel, (i+1)*n_parallel)});
        auto hard_bool_matrices = hard_matrix.unsqueeze(1) > thres.unsqueeze(-1).unsqueeze(-1);
        const int n_instances = hard_bool_matrices.size(1);
        auto hard_bool_matrices_flatten = hard_bool_matrices.reshape({-1, max_comp_per_hard_batch, max_comp_per_hard_batch});
        auto hard_length_flatten = hard_length.unsqueeze(-1).expand({hard_batch_size, n_instances}).reshape(-1).contiguous(); // [b*n]
        auto hard_comp_id_x = torch::zeros({hard_batch_size * n_instances, max_comp_per_hard_batch}, int_options);
        auto hard_comp_check = torch::ones({hard_batch_size * n_instances, max_comp_per_hard_batch}, bool_options);

        AT_DISPATCH_INDEX_TYPES(dis_length.scalar_type(), "kernel_fan_in", ([&] {
          kernel_fan_in<index_t><<<hard_batch_size * n_instances, max_comp_per_hard_batch>>>(
            hard_bool_matrices_flatten.packed_accessor32<bool,3,torch::RestrictPtrTraits>(),
            hard_length_flatten.packed_accessor32<index_t,1,torch::RestrictPtrTraits>(),
            hard_comp_id_x.packed_accessor32<index_t,2,torch::RestrictPtrTraits>(),
            hard_comp_check.packed_accessor32<bool,2,torch::RestrictPtrTraits>());
        }));

        auto hard_check = hard_comp_check.reshape({hard_batch_size, n_instances, max_comp_per_hard_batch}).all(-1); // [b, i]
        auto hard_check_batch = hard_check.any(-1);
        auto next_hard_batch = hard_check_batch.logical_not(); hbs = next_hard_batch.sum();
        const int next_hard_batch_size = long_index ? hbs.item<long>() : hbs.item<int>();

        if (next_hard_batch_size == hard_batch_size) // not found a single solution
          trials += torch::zeros_like(trials).index_put_({batch_idx}, n_instances);
        else { // find solutions to some batches
          if (long_index)
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(hard_matrix.scalar_type(), "kernel_write_choice", ([&] {
              kernel_write_choice<long, scalar_t><<<hard_batch_size, 1>>>(
                hard_check.packed_accessor32<bool,2,torch::RestrictPtrTraits>(),
                batch_idx.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
                thres.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                final_thresholds.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
                trials.packed_accessor32<long,1,torch::RestrictPtrTraits>());
            }));
          else
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(hard_matrix.scalar_type(), "kernel_write_choice", ([&] {
              kernel_write_choice<int, scalar_t><<<hard_batch_size, 1>>>(
                hard_check.packed_accessor32<bool,2,torch::RestrictPtrTraits>(),
                batch_idx.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                thres.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                final_thresholds.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
                trials.packed_accessor32<int,1,torch::RestrictPtrTraits>());
            }));
          auto hard_choice = hard_check.nonzero();
          hard_comp_id_x = hard_comp_id_x.view({hard_batch_size, n_instances, max_comp_per_hard_batch});

          AT_DISPATCH_INDEX_TYPES(dis_length.scalar_type(), "kernel_write_back", ([&] {
            kernel_write_back<index_t><<<dim3(hard_choice.size(0), max_comp_per_hard_batch), 1>>>(
              hard_choice.packed_accessor32<index_t,2,torch::RestrictPtrTraits>(),
              hard_bool_matrices.packed_accessor32<bool,4,torch::RestrictPtrTraits>(),
              hard_comp_id_x.packed_accessor32<index_t,3,torch::RestrictPtrTraits>(),
              hard_length.packed_accessor32<index_t,1,torch::RestrictPtrTraits>(),
              batch_idx.packed_accessor32<index_t,1,torch::RestrictPtrTraits>(),
              bool_matrix_x.packed_accessor32<bool,3,torch::RestrictPtrTraits>(),
              comp_id_x.packed_accessor32<index_t,2,torch::RestrictPtrTraits>(),
              comp_check.packed_accessor32<bool,2,torch::RestrictPtrTraits>());
          }));
          if (next_hard_batch_size == 0) break;

          batch_idx = batch_idx.index({next_hard_batch});
          hard_length = hard_length.index({next_hard_batch}); hlm = hard_length.max();
          seq_threshold = seq_threshold.index({next_hard_batch});
          hard_batch_size = next_hard_batch_size;
          max_comp_per_hard_batch = long_index ? hlm.item<long>() : hlm.item<int>();
          hard_batch_square_dim3 = dim3(hard_batch_size, max_comp_per_hard_batch, max_comp_per_hard_batch);
          hard_slice = torch::indexing::Slice(0, max_comp_per_hard_batch);
          hard_matrix = hard_matrix.index({next_hard_batch, hard_slice, hard_slice});
        }
      }
    }

    hard_batch = comp_check.all(1);
    if (long_index)
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(final_thresholds.scalar_type(), "kernel_write_errors", ([&] {
        kernel_write_errors<long, scalar_t><<<batch_size, 1>>>(
          hard_batch.packed_accessor32<bool,1,torch::RestrictPtrTraits>(),
          trials.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
          final_thresholds.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>());
      }));
    else
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(final_thresholds.scalar_type(), "kernel_write_errors", ([&] {
        kernel_write_errors<int, scalar_t><<<batch_size, 1>>>(
          hard_batch.packed_accessor32<bool,1,torch::RestrictPtrTraits>(),
          trials.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
          final_thresholds.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>());
      }));

    if (group_all_invalid)
      AT_DISPATCH_INDEX_TYPES(dis_length.scalar_type(), "kernel_catch_invalid_all", ([&] {
        kernel_catch_invalid_all<index_t><<<batch_square_dim3, 1>>>(
            comp_check.packed_accessor32<bool,2,torch::RestrictPtrTraits>(),
            dis_length.packed_accessor32<index_t,1,torch::RestrictPtrTraits>(),
            comp_id_x.packed_accessor32<index_t,2,torch::RestrictPtrTraits>(),
            bool_matrix_x.packed_accessor32<bool,3,torch::RestrictPtrTraits>());
      }));
    else
      AT_DISPATCH_INDEX_TYPES(dis_length.scalar_type(), "kernel_catch_invalid_each", ([&] {
        kernel_catch_invalid_each<index_t><<<batch_square_dim3, 1>>>(
            comp_check.packed_accessor32<bool,2,torch::RestrictPtrTraits>(),
            dis_length.packed_accessor32<index_t,1,torch::RestrictPtrTraits>(),
            comp_id_x.packed_accessor32<index_t,2,torch::RestrictPtrTraits>(),
            bool_matrix_x.packed_accessor32<bool,3,torch::RestrictPtrTraits>());
      }));
  }

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

  return {sections, comp_id_y, components, comp_check, final_thresholds, trials};
}