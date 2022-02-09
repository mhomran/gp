def undistort(img, k1, map_x=None, map_y=None):
  if map_x is not None and map_y is not None:
    new_img = cv.remap(img, map_x, map_y, cv.INTER_LINEAR)
    return new_img

  r = lambda x_d, y_d, c_x, c_y: math.sqrt(((x_d - c_x) ** 2) + ((y_d - c_y) ** 2))
  x_img = lambda x_d, c_x, r, k_1: c_x + (1 + k_1*r)*(x_d - c_x)
  y_img = lambda y_d, c_y, r, k_1: c_y + (1 + k_1*r)*(y_d - c_y)

  h, w, _ = img.shape
  c_x = w//2
  c_y = h//2
  new_img = np.zeros_like(img)
  map_x = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
  map_y = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)  
  for x_d in range(w):
    for y_d in range(h):
      new_x = x_img(x_d, c_x, r(x_d, y_d, c_x, c_y), k1)
      new_y = y_img(y_d, c_y, r(x_d, y_d, c_x, c_y), k1)
      
      map_x[y_d, x_d] = new_x
      map_y[y_d, x_d] = new_y
  
  new_img = cv.remap(img, map_x, map_y, cv.INTER_LINEAR)

  return new_img, map_x, map_y