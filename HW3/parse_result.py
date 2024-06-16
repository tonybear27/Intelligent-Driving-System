import json
import argparse


def parse_result(args):
  f = open(args.checkpoints)
  checkpoints = json.load(f)
  if checkpoints['progress'][0] != checkpoints['progress'][1]:
    print("evaluation not complete")
    return
  
  records = checkpoints['records']
  auto_record = {}
  plant_record = {}
  roach_record = {}
  sensor_record = {}
  new_global = {}
  for record in records:
    id = int(record['Index'])%7
    
    # interaction with autopilot
    if id < 2:
      auto_record['Success Rate'] = auto_record.get('Success Rate', 0) + int(record['Success'])
      auto_record['Collision Rate'] = auto_record.get('Collision Rate', 0) + int(record['Collisions'])
      auto_record['Avg Completion Time'] = auto_record.get('Avg Completion Time', 0) + float(record['Completion Time'])
    elif id < 4:
      plant_record['Success Rate'] = plant_record.get('Success Rate', 0) + int(record['Success'])
      plant_record['Collision Rate'] = plant_record.get('Collision Rate', 0) + int(record['Collisions'])
      plant_record['Avg Completion Time'] = plant_record.get('Avg Completion Time', 0) + float(record['Completion Time'])
    
    elif id < 6:
      roach_record['Success Rate'] = roach_record.get('Success Rate', 0) + int(record['Success'])
      roach_record['Collision Rate'] = roach_record.get('Collision Rate', 0) + int(record['Collisions'])
      roach_record['Avg Completion Time'] = roach_record.get('Avg Completion Time', 0) + float(record['Completion Time'])
      
    else:
      sensor_record['Success Rate'] = sensor_record.get('Success Rate', 0) + int(record['Success'])
      sensor_record['Collision Rate'] = sensor_record.get('Collision Rate', 0) + int(record['Collisions'])
      sensor_record['Avg Completion Time'] = sensor_record.get('Avg Completion Time', 0) + float(record['Completion Time'])
    
    if id in [0,2,4,6,7]:
      new_global['Success Rate'] = new_global.get('Success Rate', 0) + int(record['Success'])
      new_global['Collision Rate'] = new_global.get('Collision Rate', 0) + int(record['Collisions'])
      new_global['Avg Completion Time'] = new_global.get('Avg Completion Time', 0) + float(record['Completion Time'])

      
  total = len(records)/7
  auto_record['Success Rate'] = round(100*auto_record['Success Rate']/(total*2), 2)
  auto_record['Collision Rate'] = round(auto_record['Collision Rate']/(total*2), 2)
  auto_record['Avg Completion Time'] = round(auto_record['Avg Completion Time']/(total*2), 2)

  plant_record['Success Rate'] = round(100*plant_record['Success Rate']/(total*2), 2)
  plant_record['Collision Rate'] = round(plant_record['Collision Rate']/(total*2), 2)
  plant_record['Avg Completion Time'] = round(plant_record['Avg Completion Time']/(total*2), 2)

  roach_record['Success Rate'] = round(100*roach_record['Success Rate']/(total*2), 2)
  roach_record['Collision Rate'] = round(roach_record['Collision Rate']/(total*2), 2)
  roach_record['Avg Completion Time'] = round(roach_record['Avg Completion Time']/(total*2), 2)

  sensor_record['Success Rate'] = round(100*sensor_record['Success Rate']/total, 2)
  sensor_record['Collision Rate'] = round(sensor_record['Collision Rate']/total, 2)
  sensor_record['Avg Completion Time'] = round(sensor_record['Avg Completion Time']/total, 2)
  
  new_global['Success Rate'] = round(100*new_global['Success Rate']/(total*4), 2)
  new_global['Collision Rate'] = round(new_global['Collision Rate']/(total*4), 2)
  new_global['Avg Completion Time'] = round(new_global['Avg Completion Time']/(total*4), 2)
  
  checkpoints['auto record'] = auto_record
  checkpoints['plant record'] = plant_record
  checkpoints['roach record'] = roach_record
  checkpoints['sensor record'] = sensor_record
  checkpoints['new_global'] = new_global
  
  
  with open(args.checkpoints, 'w') as fd:
			json.dump(checkpoints, fd, indent=2, sort_keys=True)
  
  
if __name__ == '__main__':
  argparser = argparse.ArgumentParser(
		description='Interaction Benchmark')
  argparser.add_argument(
		'--checkpoints',
		default='./result/interaction.json',
		help='path to result file)')
  
  args = argparser.parse_args()
  parse_result(args)