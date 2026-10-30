/* Shared scalar optimizer. Moments, bias powers and clock return in safetensors. */
static double optimizer_m[PARAMETERS],optimizer_v[PARAMETERS],optimizer_powers[2];
static long optimizer_clock;
static void optimizer_reset(){
    optimizer_clock=optimizer_initial_clock;
    optimizer_powers[0]=optimizer_initial_powers[0];optimizer_powers[1]=optimizer_initial_powers[1];
    for(int i=0;i<PARAMETERS;i++){optimizer_m[i]=optimizer_initial_m[i];optimizer_v[i]=optimizer_initial_v[i];}
}
static void optimizer_apply(const double*gradient,double scale){
    if(OPTIMIZER_ADAM){optimizer_clock++;optimizer_powers[0]*=.9;optimizer_powers[1]*=.999;}
    int index=0;
    for(const auto&p:parameters)for(int j=0;j<p.size;j++,index++){
        double g=scale*gradient[index];
        if(OPTIMIZER_ADAM){
            optimizer_m[index]+=.1*(g-optimizer_m[index]);
            optimizer_v[index]=.999*optimizer_v[index]+.001*g*g;
            double mean=optimizer_m[index]/(1-optimizer_powers[0]);
            double variance=optimizer_v[index]/(1-optimizer_powers[1]);
            p.raw[j]-=LEARNING_RATE*mean/(raw_sq(variance)+1e-8);
        }else p.raw[j]-=LEARNING_RATE*g;
    }
}
extern "C" double vgi_optimizer_observe(long kind,long index){
    if(kind==0)return (double)optimizer_clock;
    if(kind==1&&index>=0&&index<PARAMETERS)return optimizer_m[index];
    if(kind==2&&index>=0&&index<PARAMETERS)return optimizer_v[index];
    if(kind==3&&index>=0&&index<2)return optimizer_powers[index];return -1;
}
