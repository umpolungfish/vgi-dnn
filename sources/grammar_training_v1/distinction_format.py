"""Typed policy transport shared by training, execution, and replay."""
ACTIONS=list(range(261,265))
VOCABULARY=299
OPERATIONS=list('≻≺⋈⊤⊙⊥⊞⊡')


def tokens(row):
    for field in ('seed_bits','positive_return'):
        if type(row[field]) is not int or not 0<=row[field]<16:
            raise ValueError(field+' must be a sixteen-state mask')
    negative=row['negative_return']
    if negative is not None and (type(negative) is not int or not 0<=negative<16):
        raise ValueError('negative_return must be a mask or None')
    if type(row['previous_action']) is not int or not 0<=row['previous_action']<4:
        raise ValueError('previous_action outside candidate registry')
    if row['requested_view'] not in (None,0,2,3):raise ValueError('unknown view criterion')
    index=row['work_index']
    if type(index) is not int or not 0<=index<len(OPERATIONS) or OPERATIONS[index]!=row['work']:
        raise ValueError('work index and bound opcode disagree')
    return [256,266+row['seed_bits'],282+index,266+row['positive_return'],
            298 if negative is None else 266+negative,290+row['previous_action'],
            {None:294,0:295,2:296,3:297}[row['requested_view']],257]


def words(sequence):
    if len(sequence)!=8 or any(type(t) is not int or not 0<=t<512 for t in sequence):
        raise ValueError('expected eight nine-bit token IDs')
    return tuple(sum(t<<(9*i) for i,t in enumerate(sequence[start:start+4])) for start in (0,4))
