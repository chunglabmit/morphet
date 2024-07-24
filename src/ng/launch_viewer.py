import neuroglancer as ng
ng.set_server_bind_address('0.0.0.0')
v = ng.Viewer()
print(v)

