/* Continuous paired memory and carrier evidence on the existing ported graph.
 * Discrete observations remain alongside the differentiable carried state.
 */
#define COMPOSED_MAX_EDGES 128
static double composed_memory[COMPOSED_MAX_EDGES][2][D],composed_probability[COMPOSED_MAX_EDGES][4];
static long composed_source_edges[COMPOSED_MAX_EDGES],composed_source_nodes[GRAPH_MAX_NODES];
static long composed_source_return,composed_passes,composed_updates,composed_started,composed_detach;
static long composed_local_targets[COMPOSED_MAX_EDGES],composed_query_target,composed_queries;
static unsigned long composed_binding;
static long composed_peak_nodes;
static double composed_loss,composed_local_loss,composed_terminal_loss;
static Var composed_mt[COMPOSED_MAX_EDGES][D],composed_mf[COMPOSED_MAX_EDGES][D];
static Var composed_p[COMPOSED_MAX_EDGES][4];
static double composed_memory_gradient[COMPOSED_MAX_EDGES][2][D];
#define COMPOSED_MAX_EVENTS (COMPOSED_MAX_EDGES*16)
static long composed_trace_count,composed_record_events;
static long composed_trace_info[COMPOSED_MAX_EVENTS][4];
static int composed_trace_ids[COMPOSED_MAX_EVENTS][2][D],composed_trace_probability_ids[COMPOSED_MAX_EVENTS][4];
static double composed_trace_memory[COMPOSED_MAX_EVENTS][2][D],composed_trace_gradient[COMPOSED_MAX_EVENTS][2][D];
static double composed_trace_probability[COMPOSED_MAX_EVENTS][4],composed_trace_probability_gradient[COMPOSED_MAX_EVENTS][4];
static unsigned long composed_trace_model;
static long composed_source_diagnostics[14],composed_total_passes;
static Var composed_join(Var a,Var b){
    if(a.v>b.v)return a;if(b.v>a.v)return b;
    return node(a.v,a.id,b.id,.5,.5);
}
static unsigned long composed_graph_binding(){
    unsigned long binding=1469598103934665603UL;
    auto bind=[&](long value){binding=(binding^(unsigned long)value)*1099511628211UL;};
    bind(graph_nodes);bind(graph_edges);bind(graph_readout);bind(graph_seed);
    for(long i=0;i<graph_nodes;i++)bind(graph_op[i]);
    for(long i=0;i<graph_edges;i++){bind(graph_from[i]);bind(graph_to[i]);bind(graph_slot[i]);bind(graph_input_slot[i]);}
    return binding;
}
static long composed_bits(const Var*p){long value=0;for(int k=0;k<4;k++)value|=(p[k].v>.5)<<k;return value;}
static Var composed_bit_loss(Var probability,long target){
    Var z=lg(probability);Var soft=z.v>0?z+ln(1+ex(-z)):ln(1+ex(z));return soft-target*z;
}
static Var composed_query(long opcode,long input,long seed,long slot,long fan,Var*mt,Var*mf,Var*probability){
    unsigned short query[8]={256,71,(unsigned short)opcode,(unsigned short)(266+input),
                            (unsigned short)(266+seed),(unsigned short)fan,(unsigned short)slot,257};
    for(int i=0;i<8;i++)retained_bank[i]=query[i];
    Evidence x,carried;
    for(int c=0;c<D;c++){x.t[0][c]=mt[c];x.f[0][c]=mf[c];}
    for(int i=1;i<TOKENS;i++)for(int c=0;c<D;c++){
        int token=i<=8?query[i-1]:0;unsigned long pos=(i-1)%MAXLEN;
        x.t[i][c]=sig(embedding_t[token*D+c]+.5*position[pos*D+c]);
        x.f[i][c]=sig(embedding_f[token*D+c]+.5*position[((pos+7)%MAXLEN)*D+c]);
    }
    for(int layer=0;layer<LAYERS;layer++)block(&x,layer);
    for(int i=0;i<TOKENS;i++)for(int c=0;c<D;c++){carried.t[i][c]=mt[c];carried.f[i][c]=mf[c];}
    residual(&carried,&x,blocks[LAYERS-1].gate1);
    Var sum=0,a[D],b[D];
    for(int c=0;c<D;c++){
        mt[c]=carried.t[8][c];mf[c]=carried.f[8][c];a[c]=lg(mt[c]);b[c]=lg(mf[c]);
        sum+=a[c]*a[c]+b[c]*b[c];
    }
    Var inv=1/sq(sum/(2*D)+1e-8);
    for(int c=0;c<D;c++){a[c]=lg(sig(final_norm[c]*a[c]*inv));b[c]=lg(sig(final_norm[c]*b[c]*inv));}
    Var support[2],refute[2];linear(headA+259*D,headB+259*D,headb+259,a,b,D,2,support,refute);
    long expected=ig_flow_gate(opcode,input,seed,slot,fan);Var loss=0;
    composed_query_target=expected;composed_queries++;
    for(int k=0;k<4;k++){
        Var z=k%2?refute[k/2]:support[k/2];probability[k]=sig(z);
        Var soft=z.v>0?z+ln(1+ex(-z)):ln(1+ex(z));loss+=soft-((expected>>k)&1)*z;
    }
    retained_prefix=8;retained_target=0;retained_kind=2;retained_bits=expected;retained_position=8;
    retained_chunks=1;retained_active=retained_complete=1;retained_window=1;
    for(int c=0;c<D;c++){retained_memory_t[c]=mt[c].v;retained_memory_f[c]=mf[c].v;}
    return loss/4;
}
static void composed_input(long node_index,Var*mt,Var*mf,Var*p){
    int found=0;
    for(long edge=0;edge<graph_edges;edge++)if(graph_to[edge]==node_index){
        for(int c=0;c<D;c++){
            if(!found){mt[c]=composed_mt[edge][c];mf[c]=composed_mf[edge][c];}
            else{mt[c]=composed_join(mt[c],composed_mt[edge][c]);mf[c]=composed_join(mf[c],composed_mf[edge][c]);}
        }
        for(int k=0;k<4;k++)p[k]=found?composed_join(p[k],composed_p[edge][k]):composed_p[edge][k];
        found=1;
    }
    if(!found){for(int c=0;c<D;c++)mt[c]=mf[c]=Var(.05);for(int k=0;k<4;k++)p[k]=Var(0);}
}
static Var composed_forward(long passes,long restart,long detach,double local_weight){
    for(long edge=0;edge<graph_edges;edge++){
        for(int c=0;c<D;c++){
            composed_mt[edge][c]=Var(restart?.05:composed_memory[edge][0][c]);
            composed_mf[edge][c]=Var(restart?.05:composed_memory[edge][1][c]);
        }
        for(int k=0;k<4;k++)composed_p[edge][k]=Var(restart?0:composed_probability[edge][k]);
    }
    Var local=0,terminal=0;long queries=0;
    for(long pass=0;pass<passes;pass++){
        for(long node_index=0;node_index<graph_nodes;node_index++){
            Var input_t[D],input_f[D],input_p[4];composed_input(node_index,input_t,input_f,input_p);
            long bits=composed_bits(input_p);graph_in[node_index]=bits;
            if(node_index==graph_readout){terminal=0;for(int k=0;k<4;k++)terminal+=composed_bit_loss(input_p[k],(composed_source_return>>k)&1);}
            for(long edge=0;edge<graph_edges;edge++)if(graph_from[edge]==node_index){
                Var mt[D],mf[D];for(int c=0;c<D;c++){
                    mt[c]=detach?Var(input_t[c].v):input_t[c];mf[c]=detach?Var(input_f[c].v):input_f[c];
                }
                local+=composed_query(graph_op[node_index],bits,graph_seed,graph_slot[edge],graph_outdegree[node_index],mt,mf,composed_p[edge]);
                composed_local_targets[edge]=composed_query_target;
                for(int c=0;c<D;c++){composed_mt[edge][c]=mt[c];composed_mf[edge][c]=mf[c];}
                if(composed_record_events){
                    long event=composed_trace_count++;if(event>=COMPOSED_MAX_EVENTS){overflowed=1;return Var(-1);}
                    composed_trace_info[event][0]=pass;composed_trace_info[event][1]=edge;
                    composed_trace_info[event][2]=bits;composed_trace_info[event][3]=composed_query_target;
                    for(int c=0;c<D;c++){
                        composed_trace_memory[event][0][c]=mt[c].v;composed_trace_memory[event][1][c]=mf[c].v;
                        composed_trace_ids[event][0][c]=mt[c].id;composed_trace_ids[event][1][c]=mf[c].id;
                        composed_trace_gradient[event][0][c]=composed_trace_gradient[event][1][c]=0;
                    }
                    for(int k=0;k<4;k++){
                        composed_trace_probability[event][k]=composed_p[edge][k].v;
                        composed_trace_probability_ids[event][k]=composed_p[edge][k].id;
                        composed_trace_probability_gradient[event][k]=0;
                    }
                }
                graph_value[edge]=composed_bits(composed_p[edge]);queries++;
                if(overflowed)return Var(-1);
            }
        }
        composed_total_passes++;
    }
    composed_local_loss=queries?local.v/queries:0;composed_terminal_loss=terminal.v/4;
    return terminal/4+(queries?local_weight*local/queries:Var(0));
}
extern "C" long vgi_composed_run(long passes,long steps,long local_millionths,long detach,long restart){
    if(!parameters_initialized)vgi_retained_reset();
    if(graph_edges>COMPOSED_MAX_EDGES||passes<1||passes>16||steps<0||steps>100||
       local_millionths<0||local_millionths>100000000||detach<0||detach>1||restart<0||restart>1)return -130;
    if(!composed_started&&!restart)return -131;
    unsigned long binding=composed_graph_binding();
    if(!restart&&binding!=composed_binding)return -135;
    composed_binding=binding;composed_queries=0;composed_peak_nodes=0;
    for(long edge=0;edge<graph_edges;edge++)for(int c=0;c<D;c++)composed_memory_gradient[edge][0][c]=composed_memory_gradient[edge][1][c]=0;
    if(restart){
        long source=(graph_started&&graph_mode==1)?graph_in[graph_readout]:vgi_graph_run(graph_seed,1,0,0,0);if(source<0)return source;
        composed_source_return=source;
        for(int i=0;i<14;i++)composed_source_diagnostics[i]=graph_diagnostics[i];
        composed_total_passes=0;
        for(long i=0;i<graph_edges;i++)composed_source_edges[i]=graph_value[i];
        for(long i=0;i<graph_nodes;i++)composed_source_nodes[i]=graph_in[i];
    }
    long original_records=retained_records;composed_updates=0;
    for(long step=0;step<(steps?steps:1);step++){
        composed_record_events=1;composed_trace_count=0;composed_trace_model=checking_model_fingerprint();
        if(steps)prepare();else retained_prepare(0);
        Var loss=composed_forward(passes,restart,detach,local_millionths/1000000.);
        if(used>composed_peak_nodes)composed_peak_nodes=used;
        if(overflowed)return -132;if(!finite_number(loss.v)||loss.v<0)return -133;
        composed_loss=loss.v;
        if(step==0)for(long edge=0;edge<graph_edges;edge++)graph_before[edge]=graph_value[edge];
        if(steps){
            backward(loss);double norm2=0;int i=0;
            for(long event=0;event<composed_trace_count;event++){
                for(int channel=0;channel<2;channel++)for(int c=0;c<D;c++){
                    int id=composed_trace_ids[event][channel][c];
                    composed_trace_gradient[event][channel][c]=id>=0?tape[id].grad:0;
                }
                for(int k=0;k<4;k++){int id=composed_trace_probability_ids[event][k];composed_trace_probability_gradient[event][k]=id>=0?tape[id].grad:0;}
            }
            for(long edge=0;edge<graph_edges;edge++)for(int c=0;c<D;c++){
                int a=composed_mt[edge][c].id,b=composed_mf[edge][c].id;
                composed_memory_gradient[edge][0][c]=a>=0?tape[a].grad:0;
                composed_memory_gradient[edge][1][c]=b>=0?tape[b].grad:0;
            }
            for(const auto&p:parameters)for(int j=0;j<p.size;j++,i++){
                initial_gradient[i]=tape[p.var[j].id].grad;
                if(!finite_number(initial_gradient[i]))return -134;norm2+=initial_gradient[i]*initial_gradient[i];
            }
            if(!finite_number(norm2))return -134;
            double norm=raw_sq(norm2),scale=(STREAM_CLIP>0&&norm>STREAM_CLIP)?STREAM_CLIP/norm:1;
            optimizer_apply(initial_gradient,scale);composed_updates++;
        }
    }
    if(steps){
        composed_record_events=0;retained_prepare(0);Var after=composed_forward(passes,restart,detach,local_millionths/1000000.);
        if(!finite_number(after.v)||after.v<0)return -133;composed_loss=after.v;
    }
    for(long edge=0;edge<graph_edges;edge++){
        for(int c=0;c<D;c++){composed_memory[edge][0][c]=composed_mt[edge][c].v;composed_memory[edge][1][c]=composed_mf[edge][c].v;}
        for(int k=0;k<4;k++)composed_probability[edge][k]=composed_p[edge][k].v;
    }
    composed_started=1;composed_passes=passes;composed_detach=detach;
    for(long edge=0;edge<graph_edges;edge++){graph_after[edge]=graph_value[edge];graph_expected[edge]=composed_source_edges[edge];}
    retained_records=original_records+1;retained_committed=composed_updates;retained_requested=steps;
    retained_peak=composed_peak_nodes;retained_recomputed=0;
    graph_mode=4;for(int i=0;i<14;i++)graph_diagnostics[i]=0;
    graph_diagnostics[1]=passes;graph_diagnostics[2]=composed_queries;
    for(long edge=0;edge<graph_edges;edge++){graph_diagnostics[3]+=graph_before[edge]!=composed_source_edges[edge];graph_diagnostics[4]+=graph_after[edge]!=composed_source_edges[edge];}
    graph_diagnostics[5]=composed_updates;graph_diagnostics[6]=4;graph_diagnostics[7]=graph_seed;
    graph_diagnostics[8]=graph_readout;graph_diagnostics[9]=passes;graph_diagnostics[11]=graph_in[graph_readout];
    graph_total_passes=composed_total_passes;graph_diagnostics[12]=composed_total_passes;graph_diagnostics[13]=parameter_digest();
    return graph_in[graph_readout];
}
extern "C" double vgi_composed_observe(long kind,long edge,long channel,long component){
    if(kind==0){if(component==0)return composed_loss;if(component==1)return composed_local_loss;if(component==2)return composed_terminal_loss;if(component==3)return composed_queries;if(component==4)return composed_peak_nodes;if(component==5)return composed_oracle_calls;if(component==6)return composed_oracle_invocations;return -1;}
    if(edge<0||edge>=graph_edges)return -1;
    if(kind==1&&component>=0&&component<4)return composed_probability[edge][component];
    if(kind==2&&channel>=0&&channel<2&&component>=0&&component<D)return composed_memory[edge][channel][component];return -1;
}
extern "C" double vgi_composed_adjoint(long edge,long channel,long component){
    if(edge<0||edge>=graph_edges||channel<0||channel>1||component<0||component>=D)return 0;
    return composed_memory_gradient[edge][channel][component];
}
extern "C" double vgi_composed_trace_observe(long kind,long event,long channel,long component){
    if(kind==0)return composed_trace_count;
    if(event<0||event>=composed_trace_count)return -1;
    if(kind==1&&component>=0&&component<4)return composed_trace_info[event][component];
    if(kind==2&&channel>=0&&channel<2&&component>=0&&component<D)return composed_trace_memory[event][channel][component];
    if(kind==3&&channel>=0&&channel<2&&component>=0&&component<D)return composed_trace_gradient[event][channel][component];
    if(kind==4&&component>=0&&component<4)return composed_trace_probability[event][component];
    if(kind==5&&component>=0&&component<4)return composed_trace_probability_gradient[event][component];return -1;
}
static unsigned long composed_checksum(unsigned long model,unsigned long retained){
    unsigned long hash=14695981039346656037UL;
    auto value=[&](unsigned long word){hash=retained_mix(hash,word);};
    auto real=[&](double x){value(retained_double_word(x));};
    value(model);value(retained);value(composed_graph_binding());
    value(graph_mode);value(graph_started);for(int i=0;i<14;i++)value(graph_diagnostics[i]);
    for(long i=0;i<graph_nodes;i++)value(graph_in[i]);
    for(long i=0;i<graph_edges;i++){value(graph_value[i]);value(graph_before[i]);value(graph_after[i]);value(graph_expected[i]);}
    value(composed_source_return);value(composed_passes);value(composed_updates);value(composed_detach);value(composed_started);value(composed_queries);value(composed_binding);
    real(composed_loss);real(composed_local_loss);real(composed_terminal_loss);
    for(long i=0;i<graph_nodes;i++)value(composed_source_nodes[i]);
    for(long i=0;i<graph_edges;i++){
        value(composed_source_edges[i]);value(composed_local_targets[i]);
        for(int k=0;k<4;k++)real(composed_probability[i][k]);
        for(int channel=0;channel<2;channel++)for(int c=0;c<D;c++){real(composed_memory[i][channel][c]);real(composed_memory_gradient[i][channel][c]);}
    }
    value(composed_trace_model);value(composed_trace_count);
    for(int i=0;i<14;i++)value(composed_source_diagnostics[i]);
    for(long event=0;event<composed_trace_count;event++){
        for(int i=0;i<4;i++)value(composed_trace_info[event][i]);
        for(int channel=0;channel<2;channel++)for(int c=0;c<D;c++){real(composed_trace_memory[event][channel][c]);real(composed_trace_gradient[event][channel][c]);}
        for(int k=0;k<4;k++){real(composed_trace_probability[event][k]);real(composed_trace_probability_gradient[event][k]);}
    }
    value(composed_oracle_calls);return hash;
}
static long composed_write_state(){
    const unsigned char magic[8]={'V','G','I','C','M','P','1',0};
    long header[11]={5,graph_nodes,graph_edges,D,graph_seed,composed_source_return,composed_passes,
                     composed_updates,composed_detach,composed_started,composed_queries};
    double losses[3]={composed_loss,composed_local_loss,composed_terminal_loss};
    if(checkpoint_write(magic,8)<0||checkpoint_write(header,sizeof(header))<0||
       checkpoint_write(&composed_binding,8)<0||checkpoint_write(losses,sizeof(losses))<0)return -136;
    if(checkpoint_write(composed_source_nodes,8*graph_nodes)<0||checkpoint_write(composed_source_edges,8*graph_edges)<0||
       checkpoint_write(composed_local_targets,8*graph_edges)<0)return -136;
    for(long edge=0;edge<graph_edges;edge++)if(checkpoint_write(composed_probability[edge],4*8)<0||
        checkpoint_write(composed_memory[edge],2*D*8)<0)return -136;
    for(long edge=0;edge<graph_edges;edge++)if(checkpoint_write(composed_memory_gradient[edge],2*D*8)<0)return -136;
    if(checkpoint_write(&composed_trace_model,8)<0||checkpoint_write(&composed_trace_count,8)<0)return -136;
    for(long event=0;event<composed_trace_count;event++)if(checkpoint_write(composed_trace_info[event],4*8)<0||
        checkpoint_write(composed_trace_memory[event],2*D*8)<0||checkpoint_write(composed_trace_gradient[event],2*D*8)<0||
        checkpoint_write(composed_trace_probability[event],4*8)<0||checkpoint_write(composed_trace_probability_gradient[event],4*8)<0)return -136;
    if(checkpoint_write(composed_source_diagnostics,sizeof(composed_source_diagnostics))<0)return -136;
    unsigned long model=checking_model_fingerprint(),retained=retained_state_fingerprint();
    unsigned long footer[4]={(unsigned long)composed_oracle_calls,model,retained,composed_checksum(model,retained)};
    if(checkpoint_write(footer,sizeof(footer))<0)return -136;
    return 0;
}
extern "C" long vgi_composed_save(){if(vgi_graph_save()<0||composed_write_state()<0)return -136;return 0;}
