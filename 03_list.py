fruits = ["apple", "mango", "dragonfruit"]
fruits[1] = "blackberry"
print(fruits)

fruits.insert(1,"orange")
print(fruits)

fruits.append("cherry")
print(fruits)

thistuple= ("kiwi", "jackfruit")
fruits.extend(thistuple)
print(fruits)

fruits.pop()
print(fruits)


i = 0
while i < len(fruits):
  print(fruits[i])
  i = i + 1


fruits.sort()
print(fruits)

