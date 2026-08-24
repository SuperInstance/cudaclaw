
#pragma once

struct CRDTState;
struct CRDTCell;

__device__ bool crdt_write_cell(CRDTState* crdt, uint32_t row, uint32_t col, double value, uint64_t timestamp, uint32_t node_id);

__device__ bool crdt_delete_cell(CRDTState* crdt, uint32_t row, uint32_t col);

__device__ bool crdt_merge_conflict(CRDTState* crdt, uint32_t row, uint32_t col);

__device__ CRDTCell crdt_read_cell(CRDTState* crdt, uint32_t row, uint32_t col);
