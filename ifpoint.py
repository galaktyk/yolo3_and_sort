from shapely.geometry import Point
from shapely.geometry.polygon import Polygon




points=((1,1,1,3),(2,4,2,1),(3,1,3,4))



polygon_a = Polygon([points[0][0:2],points[0][2:4],points[1][0:2],points[1][2:4]])

polygon_b= Polygon([points[1][0:2],points[1][2:4],points[2][0:2],points[2][2:4]])


dot = Point(1.5, 2)


print(polygon_a.contains(dot))
print(polygon_b.contains(dot))

