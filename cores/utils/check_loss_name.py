import functools
name_list = [
    'dice', 'gdice', 'ce', 'focal', 'portion', 'cl', 'hd', 
    'recall', 'dist', 'navi', 'maskednavi', 'maskeddice'
]
def check(Name):
    name = Name.lower()
    if 'gen_dice' in name: name = name.repalce('gen_dice', 'gdice')
    if 'dicece' in name: name = name.replace('dicece', 'dice_ce')
    if 'dicefocal' in name: name = name.repalce('dicefocal', 'dice_focal')
    if '_cl' in name: name = name.replace('_cl', '+cl')

    name_split = [j for i in name.split('_') for j in i.split('+')]
    if all(item in name_list for item in name_split):
        return name
    else: 
        print ("Check Loss Function Name!", Name, name, name_split) 
        raise True