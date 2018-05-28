import json

with open('data.json') as file:
    data = json.load(file)

with open("model.json") as file:
    model = json.load( file )

print( data  == model )

data_layers = data['config']['layers']
model_layers = model['config']['layers']

print((data_layers == model_layers))

names = []
for layer in data_layers:
    names.append(layer['name'])

#print( len(names) )
#print( len(set( names )))
#print( [ x for n, x in enumerate( names ) if x in names[:n]] )

