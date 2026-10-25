/* Compile with the bound exported runtime directory on the include path.
 * This adapter exposes the retained learner's existing context boundary.
 */
#include "full_model.cpp"

extern "C" long vgi_organism_config(long field) {
    if (field == 0) return PARAMETERS;
    if (field == 1) return D;
    if (field == 2) return V;
    return -1;
}
extern "C" void vgi_organism_context_begin() {
    retained_context_begin(retained_memory_t, retained_memory_f);
}
extern "C" void vgi_organism_context_end() { retained_context_end(); }
