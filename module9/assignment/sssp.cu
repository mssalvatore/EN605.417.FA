#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "nvgraph.h"

/* Shortest-Source-Single-Path
 *  Find the shortest path from a source node to every other node.

Initially :
V = 6 
E = 10

Edges       W
0 -> 1    0.50
0 -> 2    0.50
2 -> 0    0.33
2 -> 1    0.33
2 -> 4    0.33
3 -> 4    0.50
3 -> 5    0.50
4 -> 3    0.50
4 -> 5    0.50
5 -> 3    1.00


Source oriented representation (CSC):
destination_offsets {0, 1, 3, 4, 6, 8, 10}
source_indices {2, 0, 2, 0, 4, 5, 2, 3, 3, 4}
W0 = {0.33, 0.50, 0.33, 0.50, 0.50, 1.00, 0.33, 0.50, 0.50, 1.00}
*/

void check_status(nvgraphStatus_t status)
{
    if ((int)status != NVGRAPH_STATUS_SUCCESS)
    {
        printf("ERROR : %d\n",status);
        exit(0);
    }
}

void setWeights(float * weights)
{
    weights[0] = 0.333333f;
    weights[1] = 0.500000f;
    weights[2] = 0.333333f;
    weights[3] = 0.500000f;
    weights[4] = 0.500000f;
    weights[5] = 1.000000f;
    weights[6] = 0.333333f;
    weights[7] = 0.500000f;
    weights[8] = 0.500000f;
    weights[9] = 0.500000f;
}

void setDestOffsets(int * destOffsets)
{
    destOffsets[0] = 0;
    destOffsets[1] = 1;
    destOffsets[2] = 3;
    destOffsets[3] = 4;
    destOffsets[4] = 6;
    destOffsets[5] = 8;
    destOffsets[6] = 10;
}

void setSourceIndices(int * sourceIndices)
{
    sourceIndices[0] = 2;
    sourceIndices[1] = 0;
    sourceIndices[2] = 2;
    sourceIndices[3] = 0;
    sourceIndices[4] = 4;
    sourceIndices[5] = 5;
    sourceIndices[6] = 2;
    sourceIndices[7] = 3;
    sourceIndices[8] = 3;
    sourceIndices[9] = 4;
}

void getAndPrintAllPathLengths(nvgraphHandle_t handle, nvgraphGraphDescr_t graph, float * vertex_data, size_t n)
{
    for (int source_vert = 0; source_vert < n; source_vert++)
    {
        // Find Single-Source-Shortest-Path
        check_status(nvgraphSssp(handle, graph, 0, &source_vert, 0));
        
        // Get and print result
        check_status(nvgraphGetVertexData(handle, graph, (void*)vertex_data, 0));

        for (int j = 0; j < n; j++)
        {
            printf("Shortest path from %d to %d is of length %f\n", source_vert, j, vertex_data[j]);
        }
        printf("\n\n");
    }

}
int main(int argc, char ** argv)
{
    const size_t  n = 6, num_edges= 10, vertex_numsets = 3, edge_numsets = 1;

    // nvgraph variables
    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    cudaDataType_t edge_dimT = CUDA_R_32F;

    // Allocate host data
    int *destination_offsets_h = (int *) malloc((n+1)*sizeof(int));
    int *source_indices_h = (int *) malloc(num_edges*sizeof(int));
    float *weights_h = (float *) malloc(num_edges*sizeof(float));
    float *vertex_data = (float *) malloc(n * sizeof(float));
    void **vertex_dim = (void **) malloc(vertex_numsets*sizeof(void*));
    cudaDataType_t *vertex_dimT = (cudaDataType_t *) malloc(vertex_numsets*sizeof(cudaDataType_t));
    nvgraphCSCTopology32I_t CSC_input = (nvgraphCSCTopology32I_t) malloc(sizeof(struct nvgraphCSCTopology32I_st));
    
    // Initialize host data
    vertex_dim[0] = (void*)vertex_data;
    vertex_dimT[0] = CUDA_R_32F;

    setWeights(weights_h);
    setDestOffsets(destination_offsets_h);
    setSourceIndices(source_indices_h);

    // Starting nvgraph
    check_status(nvgraphCreate(&handle));
    check_status(nvgraphCreateGraphDescr(handle, &graph));

    CSC_input->nvertices = n;
    CSC_input->nedges = num_edges;
    CSC_input->destination_offsets = destination_offsets_h;
    CSC_input->source_indices = source_indices_h;
    
    // Set graph connectivity and properties (tranfers)
    check_status(nvgraphSetGraphStructure(handle, graph, (void*)CSC_input, NVGRAPH_CSC_32));
    check_status(nvgraphAllocateVertexData(handle, graph, vertex_numsets, vertex_dimT));
    check_status(nvgraphAllocateEdgeData  (handle, graph, edge_numsets, &edge_dimT));
    check_status(nvgraphSetEdgeData(handle, graph, (void*)weights_h, 0));

    getAndPrintAllPathLengths(handle, graph, vertex_data, n);
    
    //Clean 
    check_status(nvgraphDestroyGraphDescr(handle, graph));
    check_status(nvgraphDestroy(handle));

    free(destination_offsets_h);
    free(source_indices_h);
    free(weights_h);
    free(vertex_data);
    free(vertex_dim);
    free(vertex_dimT);
    free(CSC_input);

    return EXIT_SUCCESS;
}
