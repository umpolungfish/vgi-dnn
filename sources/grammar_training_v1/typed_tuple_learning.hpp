/* Axis-normalized candidate scoring on the retained executing representation.
 * Generated source bindings supply typed_query, typed_values and typed_target.
 */
static double typed_scores[12][5],typed_total_gradient[PARAMETERS];
static double typed_axis_losses[12];
extern "C" long vgi_typed_axis(long axis){
    if(axis<0||axis>=12)return -201;
    prepare();retained_capture=0;
    Var mt[D],mf[D];for(int c=0;c<D;c++)mt[c]=mf[c]=Var(.05);
    int length=typed_query_lengths[axis];retained_prefix=length;
    for(int i=0;i<length;i++)retained_bank[i]=typed_queries[axis][i];
    retained_bank[length]=258;
    long labels=0;
    for(int offset=0;offset<length;offset+=TOKENS-1)
        retained_chunk(offset,length,0,0,mt,mf,&labels);
    Var scores[5];int radix=typed_radices[axis];
    for(int value=0;value<radix;value++){
        Var bt[D],bf[D];for(int c=0;c<D;c++){bt[c]=mt[c];bf[c]=mf[c];}
        retained_bank[0]=257;int size=typed_value_lengths[axis][value];
        for(int i=0;i<size;i++)retained_bank[i+1]=typed_values[axis][value][i];
        retained_prefix=1;labels=0;
        scores[value]=-retained_chunk(0,size,0,0,bt,bf,&labels);
        typed_scores[axis][value]=scores[value].v;
    }
    Var maximum=scores[0];for(int v=1;v<radix;v++)if(scores[v]>maximum)maximum=scores[v];
    Var mass=0;for(int v=0;v<radix;v++)mass+=ex(scores[v]-maximum);
    Var loss=ln(mass)+maximum-scores[typed_target[axis]];
    if(overflowed||!finite_number(loss.v))return -202;
    backward(loss);int index=0;
    for(const auto&p:parameters)for(int j=0;j<p.size;j++){
        double g=tape[p.var[j].id].grad;
        initial_gradient[index]=g;typed_total_gradient[index]+=g/12.;index++;
    }
    typed_axis_losses[axis]=loss.v;
    return (long)(loss.v*1000000);
}
extern "C" double vgi_typed_score(long axis,long value){return typed_scores[axis][value];}
extern "C" double vgi_typed_loss(long axis){return typed_axis_losses[axis];}
extern "C" long vgi_typed_update(){
    vgi_retained_reset();for(int i=0;i<PARAMETERS;i++)typed_total_gradient[i]=0;
    for(int axis=0;axis<12;axis++)if(vgi_typed_axis(axis)<0)return -203;
    double norm2=0;for(int i=0;i<PARAMETERS;i++)norm2+=typed_total_gradient[i]*typed_total_gradient[i];
    double norm=raw_sq(norm2),scale=norm>1?1/norm:1;
    optimizer_apply(typed_total_gradient,scale);return 0;
}
