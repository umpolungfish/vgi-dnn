/* Restoration uses the shared retained-state reader after validating the
 * complete graph and continuous carrier. Model tensors are baked from return.
 */
static int composed_read_exact(void*data,unsigned long size){return stream_read(data,size)==(long)size;}
static int composed_graph_read(){
    unsigned char magic[8];long header[7];
    if(!composed_read_exact(magic,8)||!composed_read_exact(header,sizeof(header)))return 0;
    const unsigned char wanted[8]={'G','G','S','T','A','T','E','1'};
    for(int i=0;i<8;i++)if(magic[i]!=wanted[i])return 0;
    if(header[0]<1||header[0]>GRAPH_MAX_NODES||header[1]<0||header[1]>COMPOSED_MAX_EDGES||
       header[2]<0||header[2]>=header[0]||header[3]<0||header[3]>15||header[4]!=4||header[5]!=1||header[6]!=14)return 0;
    graph_nodes=header[0];graph_edges=header[1];graph_readout=header[2];graph_seed=header[3];graph_mode=header[4];graph_started=header[5];
    if(!composed_read_exact(graph_diagnostics,sizeof(graph_diagnostics)))return 0;
    long*nodes[2]={graph_op,graph_in};long*edges[8]={graph_from,graph_to,graph_slot,graph_input_slot,graph_value,graph_before,graph_after,graph_expected};
    for(int i=0;i<2;i++)if(!composed_read_exact(nodes[i],8*graph_nodes))return 0;
    for(int i=0;i<8;i++)if(!composed_read_exact(edges[i],8*graph_edges))return 0;
    for(long i=0;i<graph_nodes;i++)if(graph_in[i]<0||graph_in[i]>15||graph_op[i]<0||graph_op[i]>=GRAPH_TOKEN_COUNT)return 0;
    for(long i=0;i<graph_edges;i++)for(int j=4;j<8;j++)if(edges[j][i]<0||edges[j][i]>15)return 0;
    for(long i=0;i<graph_edges;i++)if(graph_from[i]<0||graph_from[i]>=graph_nodes||graph_to[i]<0||graph_to[i]>=graph_nodes||
        graph_slot[i]<0||graph_slot[i]>2||graph_input_slot[i]<0||graph_input_slot[i]>2)return 0;
    if(graph_diagnostics[12]<0)return 0;
    graph_total_passes=composed_total_passes=graph_diagnostics[12];return graph_validate()==0;
}
static int composed_continuous_read(unsigned long*retained_fingerprint){
    unsigned char magic[8];long header[11];double losses[3];unsigned long footer[4];
    if(!composed_read_exact(magic,8)||!composed_read_exact(header,sizeof(header))||
       !composed_read_exact(&composed_binding,8)||!composed_read_exact(losses,sizeof(losses)))return 0;
    const unsigned char wanted[8]={'V','G','I','C','M','P','1',0};
    for(int i=0;i<8;i++)if(magic[i]!=wanted[i])return 0;
    if(header[0]!=5||header[1]!=graph_nodes||header[2]!=graph_edges||header[3]!=D||header[4]!=graph_seed||
       header[5]<0||header[5]>15||header[6]<1||header[6]>16||header[7]<0||header[7]>100||
       header[8]<0||header[8]>1||header[9]!=1||header[10]<0||header[10]>COMPOSED_MAX_EVENTS*101)return 0;
    composed_source_return=header[5];composed_passes=header[6];composed_updates=header[7];
    composed_detach=header[8];composed_started=header[9];composed_queries=header[10];
    for(double loss:losses)if(!finite_number(loss)||loss<0)return 0;
    composed_loss=losses[0];composed_local_loss=losses[1];composed_terminal_loss=losses[2];
    if(!composed_read_exact(composed_source_nodes,8*graph_nodes)||!composed_read_exact(composed_source_edges,8*graph_edges)||
       !composed_read_exact(composed_local_targets,8*graph_edges))return 0;
    for(long edge=0;edge<graph_edges;edge++)if(!composed_read_exact(composed_probability[edge],4*8)||
        !composed_read_exact(composed_memory[edge],2*D*8))return 0;
    for(long edge=0;edge<graph_edges;edge++)if(!composed_read_exact(composed_memory_gradient[edge],2*D*8))return 0;
    if(!composed_read_exact(&composed_trace_model,8)||!composed_read_exact(&composed_trace_count,8)||
       composed_trace_count<0||composed_trace_count>COMPOSED_MAX_EVENTS)return 0;
    for(long event=0;event<composed_trace_count;event++)if(!composed_read_exact(composed_trace_info[event],4*8)||
        !composed_read_exact(composed_trace_memory[event],2*D*8)||!composed_read_exact(composed_trace_gradient[event],2*D*8)||
        !composed_read_exact(composed_trace_probability[event],4*8)||!composed_read_exact(composed_trace_probability_gradient[event],4*8))return 0;
    if(!composed_read_exact(composed_source_diagnostics,sizeof(composed_source_diagnostics))||
       !composed_read_exact(footer,sizeof(footer))||footer[0]>0x7fffffffffffffffUL)return 0;
    composed_oracle_calls=(long)footer[0];
    if(footer[1]!=checking_model_fingerprint()||composed_binding!=composed_graph_binding()||
       footer[3]!=composed_checksum(footer[1],footer[2]))return 0;
    for(long i=0;i<graph_nodes;i++)if(composed_source_nodes[i]<0||composed_source_nodes[i]>15)return 0;
    for(long edge=0;edge<graph_edges;edge++){
        if(composed_source_edges[edge]<0||composed_source_edges[edge]>15||composed_local_targets[edge]<0||composed_local_targets[edge]>15)return 0;
        long bits=0;for(int k=0;k<4;k++){
            double value=composed_probability[edge][k];if(!finite_number(value)||value<0||value>1)return 0;bits|=(value>.5)<<k;
        }
        if(bits!=graph_value[edge])return 0;
        for(int channel=0;channel<2;channel++)for(int c=0;c<D;c++)if(!finite_number(composed_memory[edge][channel][c])||
            composed_memory[edge][channel][c]<0||composed_memory[edge][channel][c]>1||!finite_number(composed_memory_gradient[edge][channel][c]))return 0;
    }
    for(long event=0;event<composed_trace_count;event++){
        long*info=composed_trace_info[event];if(info[0]<0||info[0]>=composed_passes||info[1]<0||info[1]>=graph_edges||info[2]<0||info[2]>15||info[3]<0||info[3]>15)return 0;
        for(int channel=0;channel<2;channel++)for(int c=0;c<D;c++)if(!finite_number(composed_trace_memory[event][channel][c])||
            composed_trace_memory[event][channel][c]<0||composed_trace_memory[event][channel][c]>1||!finite_number(composed_trace_gradient[event][channel][c]))return 0;
        for(int k=0;k<4;k++)if(!finite_number(composed_trace_probability[event][k])||composed_trace_probability[event][k]<0||
            composed_trace_probability[event][k]>1||!finite_number(composed_trace_probability_gradient[event][k]))return 0;
    }
    *retained_fingerprint=footer[2];return 1;
}
extern "C" long vgi_composed_stream(long passes,long steps,long local_millionths,long detach){
    if(!parameters_initialized)vgi_retained_reset();
    if(passes<1||passes>16||steps<0||steps>100||local_millionths<0||local_millionths>100000000||detach<0||detach>1)return -138;
    composed_io_snapshot();unsigned char magic[8];
    if(!composed_read_exact(magic,8)){composed_io_rollback();return -138;}
    const unsigned char fresh[8]={'G','G','R','A','P','H','1',0},restore[8]={'C','M','R','E','E','N','T','1'};
    int is_fresh=1,is_restore=1;for(int i=0;i<8;i++){if(magic[i]!=fresh[i])is_fresh=0;if(magic[i]!=restore[i])is_restore=0;}
    if(is_fresh){
        unsigned char header[8],row[8],extra;
        if(!composed_read_exact(header,8)){composed_io_rollback();return -138;}
        long n=u16(header),e=u16(header+2),sink=u16(header+4),seed=u16(header+6);
        if(e>COMPOSED_MAX_EDGES||seed>15||vgi_graph_reset(n,e,sink)<0){composed_io_rollback();return -138;}
        for(long i=0;i<n;i++)if(!composed_read_exact(row,2)||vgi_graph_node(i,u16(row))<0){composed_io_rollback();return -138;}
        for(long i=0;i<e;i++)if(!composed_read_exact(row,8)||vgi_graph_edge(i,u16(row),u16(row+2),u16(row+4),u16(row+6))<0){composed_io_rollback();return -138;}
        if(stream_read(&extra,1)!=0||graph_validate()<0||(!e&&steps)){composed_io_rollback();return -138;}
        long result=vgi_graph_run(seed,1,0,0,0);if(result<0)return result;
        return vgi_composed_run(passes,steps,local_millionths,detach,1);
    }
    if(is_restore){
        unsigned long options[4],fingerprint;
        if(!composed_read_exact(options,sizeof(options))||options[0]>16||options[1]>100||options[2]>100000000||
           options[3]>1||(!options[0]&&options[1])||!composed_graph_read()||!composed_continuous_read(&fingerprint)){
            composed_io_rollback();return -139;
        }
        unsigned short expected[8];const unsigned short*view=0;
        long last=-1;for(long node=0;node<graph_nodes;node++)for(long edge=0;edge<graph_edges;edge++)if(graph_from[edge]==node)last=edge;
        if(last>=0){
            long node=graph_from[last];unsigned short query[8]={256,71,(unsigned short)graph_op[node],(unsigned short)(266+graph_in[node]),
                (unsigned short)(266+graph_seed),(unsigned short)graph_outdegree[node],(unsigned short)graph_slot[last],257};
            for(int i=0;i<8;i++)expected[i]=query[i];view=expected;
        }
        if(retained_read_state(view,last>=0?8:0,1,fingerprint,1)<0){composed_io_rollback();return -139;}
        composed_peak_nodes=(long)retained_peak;composed_record_events=0;
        if(!options[0])return graph_in[graph_readout];
        return vgi_composed_run((long)options[0],(long)options[1],(long)options[2],(long)options[3],0);
    }
    composed_io_rollback();return -138;
}
