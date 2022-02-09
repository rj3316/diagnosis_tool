from os import getcwd, mkdir
import shutil
from datetime import datetime
import pandas as pd
import tkinter as tk
from tkinter import filedialog as fd
import re
# pip install cufflinks
# pip install chart_studio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pickle
import logging
import time

class Diagnosis_Tool():
	def __init__(self, file = None):
		self.file = file

		# Ask for file if no file is externally set
		if self.file is None:
			self.ask_for_file()

		if (self.file is None) or (self.file == ''):
			print("Cancelled by user...")
		else:
    		# Establecemos la configuraion comun a todos los formatos
			self.configuration_common()

    		# Probamos con distintos formatos de texto
			for i in range(5):
				print(f"Trying with format {i}")

				try:
					self.format = i

					# Establecemos la configuraion particular de cada formato
					self.configuration_format()

					self.read()

					break
				except Exception as e:
					print(e)

					self.delete_output_folder()

	def configuration_common(self):
		# Config
		self.config = {}
		self.config['create_folder'] = True
		self.config['create_summary_log'] = True
		self.config['create_timeseries_log'] = True
		self.config['create_updates_log'] = True

		self.config['dump_object'] = False
		self.config['plot_hystogram'] = True
		self.config['auto_open_hystogram'] = False   		
		self.config['plot_timeseries'] = True
		self.config['auto_open_timeseries'] = False   		
		self.config['n_lines_timeseries'] = float('inf') # For all file use self.config['n_lines_timeseries'] = float('inf')

		self.save_names = {}
		self.save_names['hystogram'] = 'hystogram.html'
		self.save_names['timeseries'] = 'timeseries.html'
		self.save_names['dump_object'] = 'diagnosis_tool_dump.pkl'
		self.save_names['summary_log'] = 'summary.log'
		self.save_names['timeseries_log'] = 'timeseries.log'
		self.save_names['updates_log'] = 'updates.log'

		self.sep = {}
		self.search = {}

		# Time emulator
		self.time_emulator = '----'

		# Each section search identifier
		self.search['fw_version'] = 'Firmware name: '
		self.search['mcp_info'] = 'mcp_info :'
		
		self.search['local_time'] = 'HWClock DateTime.toString= '
		self.search['converter'] = 'ConverterName= '
		self.search['data'] = 'SoFch='

		# self.search['layout'] = 'MEC Layout: '

	def configuration_format(self):
		self.tmp_summary_log = []
		self.tmp_timeseries_log = []
		self.tmp_updates_log = []

		# Text delimiters
		if self.format == 0:
			# Newest format
			self.sep['time'] = 'Z'  
			self.sep['update'] = 'T'

			self.search['layout'] = 'MEC Layout: '
			self.search['ip'] = '"IPv4.local_address"='

			self.search['update'] = ['Error SoC', 'SoC Error']

			self.data_columns = ['time', 'line', 'sof_ch', 'sof_dsch', 'soc', 'ah_ch', 'ah_dsch', 'voltage', 'state', 'tp2', 'tp1', 'current', 'vcell_max', 'vcell_min']

			self.update_format = 1		

		elif self.format == 1:
			# Newer format
			self.sep['time'] = 'Z'  
			self.sep['update'] = 'T'

			self.search['layout'] = 'MEC Layout: '
			self.search['ip'] = '"IPv4.local_address"='

			self.search['update'] = ['Error SoC', 'SoC Error']

			self.data_columns = ['time', 'line', 'sof_ch', 'sof_dsch', 'soc', 'ah_ch', 'ah_dsch', 'voltage', 'state', 'temperature', 'current', 'vcell_max', 'vcell_min']

			self.update_format = 0

		if self.format == 2:
    			# Old format
			self.sep['time'] = '\t'
			self.sep['update'] = ' '

			self.search['layout'] = 'MECs in the battery.'  		
			self.search['ip'] = 'Local machine IP: '
			
			self.search['update'] = 'SoC error'

			self.data_columns = ['time', 'line', 'sof_ch', 'sof_dsch', 'soc', 'voltage', 'state', 'temperature', 'current', 'vcell_max', 'vcell_min']

			self.update_format = 0

	def ask_for_file(self):
		root = tk.Tk()
		root.withdraw()

		filetypes = (
			('text files', '*.txt'),
			('All files', '*.*')
		)

		file = fd.askopenfilename(title = "Select log file", initialdir = ".\\", filetypes = filetypes)

		if file == '':
			self.file = None
		else:
			self.file = file

	# Read all info
	def read(self, verbose = True):
		self.get_info()

		self.get_local_time()
		self.get_fw_version()
		self.get_mcp_info()
		self.get_ip()
		self.get_converter()
		self.get_layout()

		# TIMESERIES DATA
		self.get_global_timeseries()

		# UPDATES
		self.get_updates()

		# new
		self.create_output_folder()
		self.create_timeseries_log()
		self.create_summary_log()
		self.create_updates_log()

		self.print_file(verbose = verbose)
		self.print_local_time(verbose = verbose)
		self.print_fw_version(verbose = verbose)
		self.print_mcp_info(verbose = verbose)
		self.print_ip(verbose = verbose)
		self.print_converter(verbose = verbose)
		self.print_layout(verbose = verbose)

		self.print_global_timeseries(verbose = verbose)
		self.print_updates(verbose = verbose)

		# Plot global timeseries
		self.plot_global_timeseries()

		# Plot updates hystogram
		self.plot_updates_hystrogram(self.get_updates_hystogram())

		# Get updates timeseries
		# Plot udpates timeseries

		# Plot individual updates timeseries

		# Plot stats -> Merge with updates timeseries

		# Save object to file
		self.save_diagnosis_tool_to_file()


		message = 'DONE!\n'
		print(message)
		self.write_summary_log(message)


	def get_info(self):
		start_time = datetime.now()

		message = f"\nReading {self.file}..."
		self.tmp_summary_log.append(message)

		fid = open(self.file, 'r')
		in_data = fid.readlines()
		fid.close()

		# Check if format is correct
		if not self.sep['time'] in in_data[0]:
			raise ValueError("Error reading")

		data = []
		data.append('TIME' + self.sep['time'] + 'INFO') # 
		for line in in_data:
			if len(line.split(self.sep['time'], maxsplit = 1)) >= 2:
				new_line = line
			else:
				new_line = self.time_emulator + self.sep['time'] + line

			data.append(new_line)

		data = list(map(lambda x: x.replace('\n', ''), data))
		data = list(map(lambda x: x.split(self.sep['time'], maxsplit = 1), data))

		df = pd.DataFrame(data, columns = ['time', 'info'])
		self.df_info = df

		message = f"  Readed in {datetime.now() - start_time} seconds!"
		self.tmp_summary_log.append(message)

	# Print file
	def print_file(self, verbose = False):
		message = f"Diagnosis Tool launched on {datetime.now()}"
		message += f"\nRead file: {self.file}"

		if verbose:
			print(message)

		self.write_summary_log(message)	

		message = list()
		message.append(self.tmp_summary_log.pop(0))
		message.append(self.tmp_summary_log.pop(0))
		message = '\n'.join(message)

		self.write_summary_log(message)



	# Get local time
	def get_local_time(self):
		df = self.df_info
		
		try:
			tmp = df['info'][df['info'].str.contains(self.search['local_time'], na = False)]

			tmp = list(tmp)[0]
			tmp = tmp.replace(self.search['local_time'], '')
		except:
			tmp = "Not found"

		self.local_time = tmp

	def print_local_time(self, verbose = False):
		message = f"\nLocal time: {self.local_time}"

		if verbose:
			print(message)

		self.write_summary_log(message)

	# Get FW version	
	def get_fw_version(self):
		df = self.df_info

		try:
			tmp = df['info'][df['info'].str.contains(self.search['fw_version'], na = False)]

			tmp = list(tmp)[0]
			tmp = tmp.replace(self.search['fw_version'], '').replace(' ', '')
		except:
			tmp = "Not found"

		self.fw_version = tmp	

	def print_fw_version(self, verbose = False):
		message = f"\nFW Version: {self.fw_version}"

		if verbose:
			print(message)

		self.write_summary_log(message)	

	# Get MCP info
	def get_mcp_info(self):
		df = self.df_info

		try:
			tmp = df['info'][df['info'].str.contains(self.search['mcp_info'], na = False)]

			tmp = list(tmp)[0]
			tmp = tmp.replace(self.search['mcp_info'], '').replace(' ', '').split("|")

			tmp_name = tmp[0].replace("\"", "")
			tmp_serial = tmp[1].replace("\'", "")
		except:
			tmp_name = "Not found"
			tmp_serial = "Not found"

		self.mcp_info = {}
		self.mcp_info['name'] = tmp_name
		self.mcp_info['serial']   = tmp_serial
	
	def print_mcp_info(self, verbose = False):
		message = f"\nMCP info:\n - MCP Name: {self.mcp_info['name']}\n - MCP Serial: {self.mcp_info['serial']}"

		if verbose:
			print(message)

		self.write_summary_log(message)  		

	# Get IP
	def get_ip(self):
		df = self.df_info

		try:
			tmp = df['info'][df['info'].str.contains(self.search['ip'], na = False)]

			tmp = list(tmp)[0]
			tmp = tmp.replace(self.search['ip'], '').replace(' ', '').replace('|', '').replace('\"', '')
		except:
			tmp = "Not found"
	
		self.ip = tmp

	def print_ip(self, verbose = False):
		message = f"\nIP: {self.ip}"

		if verbose:
			print(message)

		self.write_summary_log(message)

	# Get Converter
	def get_converter(self):
		df = self.df_info

		tmp = df['info'][df['info'].str.contains(self.search['converter'], na = False)]

		tmp = list(tmp)[0]
		tmp = tmp.replace(self.search['converter'], '').replace(' ', '').replace('"', '')

		self.converter = tmp

	def print_converter(self, verbose = False):
		message = f"\nConverter: {self.converter}"

		if verbose:
			print(message)

		self.write_summary_log(message)

	# Get layout
	def get_layout(self):
		df = self.df_info

		tmp = df['info'][df['info'].str.contains(self.search['layout'], na = False)]

		index = tmp.index[0]
		index_offset = 1

		tmp = list(tmp)[0]
		tmp = tmp.replace(self.search['layout'], '').split('(')[0].replace(' ', '')

		self.n_modules = int(tmp)    

		layout = []
		for i in range(self.n_modules):
			try:
				info = df['info'].iloc[index + index_offset + i]
				info = list(map(lambda x: x.replace(' ', ''), info.split('|')))

				tmp = {}
				tmp['mec_id'] = info[0].split('=')[1]
				tmp['mec_sn'] = info[1].split('=')[1]
				tmp['mec'] = 'mec' + tmp['mec_id']

				layout.append(tmp)
			except:
				tmp = {}
				tmp['mec_id'] = i + 1
				tmp['mec_sn'] = 'NA'
				tmp['mec']    = 'mec' + str(i + 1)

				layout.append(tmp)				

		self.layout = layout

	def print_layout(self, verbose = False):
		message = f"\nModules: {self.n_modules}"
		for layout in self.layout:
			message += f"\n - MEC ID: {layout['mec_id']} -> MEC SN: {layout['mec_sn']}    -> {layout}"

		if verbose:
			print(message)

		self.write_summary_log(message)

	# Get global timeseries
	def get_global_timeseries(self, verbose = True):
		df = self.df_info

		# Inicializamos el DataFrame
		self.n_data = 0

		self.df_global_timeseries = pd.DataFrame(columns = self.data_columns)

		# Identificamos las líneas con datos
		tmp = df['info'][df['info'].str.contains(self.search['data'], na = False)]

		message = f"\nFound {len(tmp)} data lines!"
		self.tmp_summary_log.append(message)

		# Leemos los datos
		if len(tmp) > 0:
			message = f"\nReading data ({min([self.config['n_lines_timeseries'], len(tmp)])} registers will be read)..."
		else:
			message = "\nNo data plot will be shown due to no data found"

		self.tmp_summary_log.append(message)

		for it,i in enumerate(tmp.index):
    		
			if it >= self.config['n_lines_timeseries']:
				break
			
			try:
				# Volcamos al archivo de timeseries_log				
				message = str(df['time'].iloc[i]) + ' | ' + df['info'].iloc[i].replace('(INFO)  ', '')
				self.tmp_timeseries_log.append(message)

				# Mostramos por pantalla el progreso de la operación
				if (it % 500 == 0):
					message = f"  Reading line {i} ({it} of {min([self.config['n_lines_timeseries'], len(tmp)])})..."
					self.tmp_summary_log.append(message)

				# Creamos la lista con la información
				tmp_info = []

				# Añadimos el datetime
				tmp_info.append(pd.to_datetime(df['time'].iloc[i]))

				# Añadimos la línea desde la que se ha leido
				tmp_info.append(i)

				# Leemos la línea correspondiente, la limpiamos y hacemos el split
				tmp_info_values = df['info'].iloc[i].replace(" ", "").replace("(INFO)", "").split('|')

				# Hacemos split a cada elemento para separar nombre y valor
				tmp_info_values = list(map(lambda x: x.split("=")[1], tmp_info_values))
			
				# Limpiamos  valores extraños
				tmp_info_values = list(map(lambda x: re.findall(r'-?\d+\.?\d*', x)[0], tmp_info_values))

				# Convertimos todos los datos a tipo float
				tmp_info_values = list(map(lambda x: float(x), tmp_info_values))			

				# Añadimos los datos despues del datetime
				tmp_info += tmp_info_values

				# Creamos la lista con la que construiremos el DataFrame
				tmp_data = []
				tmp_data.append(tmp_info)

				new_df = pd.DataFrame(data = tmp_data, columns = self.data_columns)

				# Concatenamos los datos que ya se habían leido y los nuevos 
				self.df_global_timeseries = pd.concat([self.df_global_timeseries, new_df])

				self.n_data += 1

			except Exception as e:
				if verbose:
					message = f" - Failed reading data on line {i}: {df['info'].iloc[i]}"
					message += f"\n    {e}"
					self.tmp_summary_log.append(message)

					raise ValueError("Incorrect timeseries format")

		self.df_global_timeseries.set_index("time", inplace = True)

	def print_global_timeseries(self, verbose = False):
		for tmp in self.tmp_timeseries_log:
			self.write_timeseries_log(tmp)

		for tmp in self.tmp_summary_log:
			if verbose:
				print(tmp)

			self.write_summary_log(tmp)

	def plot_global_timeseries(self):
		if self.config['plot_timeseries']:
			# Initialize subplot
			rel = False

			if rel:
				ts_x_init = self.df_global_timeseries.index.tolist()[0]

				try:
					# ts_x = list(map(lambda x: time.gmtime(x.total_seconds()), ts_x))
					ts_x = list(map(lambda x: x - ts_x_init, self.df_global_timeseries.index.tolist()))
					ts_x = list(map(lambda tdelta: datetime.strptime(str(tdelta, "%H:%M:%S")), ts_x))
				except Exception as e:
					print(e)

			else:
				ts_x = self.df_global_timeseries.index.tolist()

			row = 0
			col = 0

			# Create subplot
			self.global_timeseries_fig = make_subplots(rows = 3, cols = 2, subplot_titles = ['State', 'Voltage', 'SoC', 'Vcell', 'Current', 'Temperature'], shared_xaxes = True)		

			# State
			row += 1
			col += 1

			ts_y = self.df_global_timeseries['state'].values
			self.global_timeseries_fig = self.add_plot_go(fig = self.global_timeseries_fig, x = ts_x, y = ts_y, row = row, col = col, type = 'line', name = 'State')

			# Voltage
			col += 1

			ts_y = self.df_global_timeseries['voltage'].values
			self.global_timeseries_fig = self.add_plot_go(fig = self.global_timeseries_fig, x = ts_x, y = ts_y, row = row, col = col, type = 'line', name = 'Voltage')

			# SoC
			row += 1
			col = 1

			ts_y = self.df_global_timeseries['soc'].values
			self.global_timeseries_fig = self.add_plot_go(fig = self.global_timeseries_fig, x = ts_x, y = ts_y, row = row, col = col, type = 'line', name = 'SoC')

			# Vcell
			col += 1

			# Vcell max
			ts_y = self.df_global_timeseries['vcell_max'].values
			self.global_timeseries_fig = self.add_plot_go(fig = self.global_timeseries_fig, x = ts_x, y = ts_y, row = row, col = col, type = 'line', name = 'Vcell MAX')

			# Vcell min
			ts_y = self.df_global_timeseries['vcell_min'].values
			self.global_timeseries_fig = self.add_plot_go(fig = self.global_timeseries_fig, x = ts_x, y = ts_y, row = row, col = col, type = 'line', name = 'Vcell MIN')			

			# Current
			row += 1
			col = 1

			ts_y = self.df_global_timeseries['current'].values
			self.global_timeseries_fig = self.add_plot_go(fig = self.global_timeseries_fig, x = ts_x, y = ts_y, row = row, col = col, type = 'line', name = 'Current')

			# SoF Charge
			ts_y = +self.df_global_timeseries['sof_ch'].values # Añadimos el signo según el convenio de signos bateria (descarga -, carga +)
			self.global_timeseries_fig = self.add_plot_go(fig = self.global_timeseries_fig, x = ts_x, y = ts_y, row = row, col = col, type = 'line', name = 'SoF Charge')

			# SoF Discharge
			ts_y = -self.df_global_timeseries['sof_dsch'].values # Añadimos el signo según el convenio de signos bateria (descarga -, carga +)
			self.global_timeseries_fig = self.add_plot_go(fig = self.global_timeseries_fig, x = ts_x, y = ts_y, row = row, col = col, type = 'line', name = 'SoF Discharge')


			# Temperatures
			col += 1

			try:
				ts_y = self.df_global_timeseries['temperature'].values
				self.global_timeseries_fig = self.add_plot_go(fig = self.global_timeseries_fig, x = ts_x, y = ts_y, row = row, col = col, type = 'line', name = 'Temperature')
			except:
				ts_y = self.df_global_timeseries['tp1'].values
				self.global_timeseries_fig = self.add_plot_go(fig = self.global_timeseries_fig, x = ts_x, y = ts_y, row = row, col = col, type = 'line', name = 'Temp min')

				ts_y = self.df_global_timeseries['tp2'].values
				self.global_timeseries_fig = self.add_plot_go(fig = self.global_timeseries_fig, x = ts_x, y = ts_y, row = row, col = col, type = 'line', name = 'Temp max')

			# Update and export
			self.global_timeseries_fig.update_layout(height = 900, width = 1600, title_text = 'TIMESERIES')

			# self.global_timeseries_fig.update_xaxes(range = [datetime(year = 2022, month = 2, day = 8, hour = 11, minute = 20),
        	#                 datetime(year = 2022, month = 2, day = 8, hour = 13, minute = 7)])	

			self.global_timeseries_fig.update_xaxes(matches='x')

			self.global_timeseries_fig.write_html(self.output_folder + '\\' + self.save_names['timeseries'], auto_open = self.config['auto_open_timeseries'])			

	# Get updates
	def get_updates(self):
		df = self.df_info

		self.n_updates = 0
		self.n_error_updates = 0

		self.updates = []
		
		if type(self.search['update']) == str:
			search = list()
			search.append(self.search['update'])
		else:
			search = self.search['update']

		for srch_upd in search:
			tmp = None
			tmp = df['info'][df['info'].str.contains(srch_upd, na = False)]

			if len(tmp) > 0:
				break		

		message = f"\nFound {len(tmp)} updates!"
		self.tmp_summary_log.append(message)

		if len(tmp) > 0:
			message = f"  Reading updates..."
		else:
			message = "  No update plot will be shown due to no update found"

		self.tmp_summary_log.append(message)

		for i in tmp.index:
			message = f"  * Reading update in line {i}"

			self.tmp_summary_log.append(message)

			try:			
				self.updates.append(soc_update(self, init = i, sep = self.sep['update']))
				self.n_updates += 1

				message = "    Succesfully read!"

				self.tmp_summary_log.append(message)

			except Exception as e:
				self.n_error_updates += 1

				message = f"    Error reading update in line {i}..."
				message += f"\n      {e}"
				self.tmp_summary_log.append(message)

		message = f" Readed {self.n_updates} updates from total of {len(tmp)} ({self.n_error_updates} failed)"
		self.tmp_summary_log.append(message)

	def print_updates(self, verbose = False):
		for tmp in self.tmp_updates_log:
			if verbose:		
				print(tmp)

			self.write_updates_log(tmp)    		

	# Get updates hystogram
	def get_updates_hystogram(self):
		# Inicializamos el conteo de updates UP y DOWN
		hystogram = {}

		# Creamos un contador para el total de los updates
		hystogram['total'] = {
			'up': 0,
			'down': 0,
		}

		for n in range(self.n_modules):
    		# Creamos un contador para cada mec
			hystogram[self.layout[n]['mec']] = {
				'up': 0,
				'down': 0,
			}

		# Recorremos los updates y vamos contando
		for upd in self.updates:
			updater_mec = upd.get_updater_mec()[0]
			update_type = upd.type

			hystogram[updater_mec][update_type] += 1
			hystogram['total'][update_type] += 1

		df_hystogram = pd.DataFrame.from_dict(hystogram)

		message = f"\nHYSTOGRAM:\n{df_hystogram}"
		self.tmp_summary_log.append(message)

		return df_hystogram

	def plot_updates_hystrogram(self, hystogram):
		if self.config['plot_hystogram']:
			df_hyst_mecs       = hystogram.columns[1:]
			df_hyst_up         = list(map(lambda x: x[0], hystogram[hystogram.index.isin(['up'])].T.values))
			df_hyst_up_total   = df_hyst_up[0]
			df_hyst_up         = df_hyst_up[1:]
			df_hyst_down       = list(map(lambda x: x[0], hystogram[hystogram.index.isin(['down'])].T.values))
			df_hyst_down_total = df_hyst_down[0]
			df_hyst_down       = df_hyst_down[1:]

			subplot_titles = [f'UP (total: {df_hyst_up_total} updates)', f'DOWN (total: {df_hyst_down_total} updates)']

			self.hystogram_fig = make_subplots(rows = 1, cols = 2, subplot_titles = subplot_titles)
			self.hystogram_fig = self.add_plot_go(fig = self.hystogram_fig, x = df_hyst_mecs, y = df_hyst_up, row = 1, col = 1, type = 'bar', name = 'Updates UP')
			self.hystogram_fig = self.add_plot_go(fig = self.hystogram_fig, x = df_hyst_mecs, y = df_hyst_down, row = 1, col = 2, type = 'bar', name = 'Updates DOWN')

			self.hystogram_fig.update_layout(height = 600, width = 1200, title_text = 'UPDATES HYSTOGRAM')
			self.hystogram_fig.write_html(self.output_folder + '\\' + self.save_names['hystogram'], auto_open = self.config['auto_open_hystogram'])			

	# Add generic plot
	def add_plot_go(self, fig, x, y, row = 1, col = 1, type = 'line', name = 'trace',font = None):
		if font is None:
			font = dict(
				family = "Courier New, monospace",
				size = 18,
				color = "RebeccaPurple"
			)    		

		if type == 'line':
			tmp_go = go.Scatter(x = x,
					y = y,
					name = name)
		elif type ==  'bar':
			tmp_go = go.Bar(x = x,
					y = y,
					name = name)
		fig.add_trace(
			tmp_go,
			row = row, 
			col = col
			)    	

		return fig

	# Create output folder
	def create_output_folder(self):
		if self.config['create_folder']:
			output_folder = 'output_' + datetime.now().strftime('%Y%m%d') + '_' + datetime.now().strftime('%H%M%S')

			filename = self.file.split('/')[-1].split('.')[0]
			output_folder += f'_F_{filename}'

			self.output_folder = output_folder

			mkdir(output_folder)

	def delete_output_folder(self):
		try:
			shutil.rmtree(self.output_folder, ignore_errors = False)
		except:
			pass

	### Log functions
	# Create timeseries log
	def create_timeseries_log(self):
		if self.config['create_timeseries_log']:
			filename = self.output_folder + '\\' + self.save_names['timeseries_log']
			# self.setup_logger(logger_name = 'timeseries_log', log_file = filename)

			# Create timeseries logger
			logger_name = 'timeseries_log'
			l = logging.getLogger(logger_name)
			formatter = logging.Formatter('%(message)s')
			fileHandler = logging.FileHandler(filename, mode='w')
			fileHandler.setFormatter(formatter)
			# StreamHandler saca por el stdout el mensaje del logger
			# streamHandler = logging.StreamHandler()
			# streamHandler.setFormatter(formatter)

			l.setLevel(logging.INFO)
			l.addHandler(fileHandler)
			# l.addHandler(streamHandler)  			

			self.timeseries_log = logging.getLogger(logger_name)
			self.timeseries_log_file = filename

	# Write timeseries log
	def write_timeseries_log(self, message):	
		if self.config['create_timeseries_log']:
			self.timeseries_log.info(message)			

	# Create summary log
	def create_summary_log(self):
		if self.config['create_summary_log']:
			filename = self.output_folder + '\\' + self.save_names['summary_log']
			# self.setup_logger('summary_log', filename)

			# Create summary logger
			logger_name = 'summary_log'
			l = logging.getLogger(logger_name)
			formatter = logging.Formatter('%(message)s')
			fileHandler = logging.FileHandler(filename, mode='w')
			fileHandler.setFormatter(formatter)
			# StreamHandler saca por el stdout el mensaje del logger
			# streamHandler = logging.StreamHandler()
			# streamHandler.setFormatter(formatter)

			l.setLevel(logging.INFO)
			l.addHandler(fileHandler)
			# l.addHandler(streamHandler)  			

			self.summary_log = logging.getLogger(logger_name)
			self.summary_log_file = filename

	# Write summary log
	def write_summary_log(self, message):
		if self.config['create_summary_log']:
			self.summary_log.info(message)    		

	# Create updates log
	def create_updates_log(self):
		if self.config['create_updates_log']:
			filename = self.output_folder + '\\' + self.save_names['updates_log']
			# self.setup_logger(logger_name = 'timeseries_log', log_file = filename)

			# Create logger
			logger_name = 'updates_log'
			l = logging.getLogger(logger_name)
			formatter = logging.Formatter('%(message)s')
			fileHandler = logging.FileHandler(filename, mode='w')
			fileHandler.setFormatter(formatter)
			# StreamHandler saca por el stdout el mensaje del logger
			# streamHandler = logging.StreamHandler()
			# streamHandler.setFormatter(formatter)

			l.setLevel(logging.INFO)
			l.addHandler(fileHandler)
			# l.addHandler(streamHandler)  			

			self.updates_log = logging.getLogger(logger_name)
			self.updates_log_file = filename

	# Write updates log
	def write_updates_log(self, message):
		if self.config['create_updates_log']:
			self.updates_log.info(message)       		

	# Dump object
	def save_diagnosis_tool_to_file(self):
		if self.config['dump_object']:
			filename = self.output_folder + '\\' + self.save_names['dump_object']

			with open(filename, 'wb') as outp:
				pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)


class soc_update():
	def __init__(self, model = None, init = 0, sep = None):
		self.model = model

		self.init_point = init 		# Integer 
		self.datetime = None		# Datetime
		self.df_voltages = None		# DataFrame -> (mec, vmin, vmax)
		self.soc = {}				# {} -> (ini, end, delta)

		# Paramos en un update en concreto
		if self.init_point == 2050:
			pause = True

		# Obtenemos la información del update
		if not self.model is None:
			self.get_datetime(sep)
			self.get_soc()
			self.get_update_type()
			self.get_voltages()

			self.print_update()

	def get_datetime(self, sep):
		model = self.model
		# Info del datetime
		try:
			aux = model.df_info['time'][self.init_point].split(sep)

			date = list(map(lambda x: round(float(x)), aux[0].split('-')))
			time = list(map(lambda x: round(float(x)), aux[1].split(':')))
			tmp_datetime = datetime(year = date[0], month = date[1], day = date[2], hour = time[0], minute = time[1], second = round(time[2]))
		except Exception as e:
			print(e)

			tmp_datetime = datetime(year = 2021, month = 1, day = 1, hour = 0, minute = 0, second = 0)

		# Volcamos la salida al objeto
		self.datetime = tmp_datetime

	def get_soc(self):
		model = self.model

		# INFO SOC
		tmp_soc = {}

		# Info sobre el SoC inicial antes del update
		soc_ini_point = self.init_point

		cont = True
		above = True
		found = False
		if above:
			while cont:
				# Buscamos la primera línea con datos (consideramos ese valor de SoC )
				if model.search['data'] in model.df_info['info'][soc_ini_point]:
					tmp_soc['ini']  = int(model.df_info['info'][soc_ini_point].split('|')[2].split('=')[1].replace('%', ''))

					cont = False
					above = False
					found = True
				else:
					# Si no encontramos una línea con datos, subimos una línea (pueden haberse escrito datos en medio del update...)
					soc_ini_point -= 1

				# Si nos alejamos X puntos hacia arriba respecto del inicio, paramos y buscamos en la otra dirección
				if abs(soc_ini_point - self.init_point) >= 20:
					above = False
					cont = False
					found = False

		if (not above) and (not found):
			cont = True
			above = False
			found = False    			
			while cont:
				# Buscamos la primera línea con datos (consideramos ese valor de SoC )
				if model.search['data'] in model.df_info['info'][soc_ini_point]:
					tmp_soc['ini']  = int(model.df_info['info'][soc_ini_point].split('|')[2].split('=')[1].replace('%', ''))

					cont = False
					above = False
					found = True
				else:
					# Si no encontramos una línea con datos, bajamos una línea (pueden haberse escrito datos en medio del update...)
					soc_ini_point += 1

				# Si nos alejamos X puntos hacia abajo respecto del inicio, paramos y consideramos que no existe la información
				if abs(soc_ini_point - self.init_point) >= 20:
					tmp_soc['ini'] = 0

					above = False
					cont = False
					found = False					

		# Info sobre el delta de SoC en el update
		# if model.update_format == 0:
		# 	tmp_soc['delta'] = int(model.df_info['info'][self.init_point].split(',')[-1].replace('%', '').replace(')', '').replace(' ', ''))
		# elif model.update_format == 1:
		# 	tmp_soc['delta'] = int(model.df_info['info'][self.init_point].split('|')[-1].replace('%', '').replace(' ', ''))
		try:
			tmp_soc['delta'] = int(model.df_info['info'][self.init_point].split(',')[-1].replace('%', '').replace(')', '').replace(' ', ''))
		except:
			tmp_soc['delta'] = int(model.df_info['info'][self.init_point].split('|')[-1].replace('%', '').replace(' ', ''))
			
		# Info sobre el SoC final despues del update
		tmp_soc['end'] = tmp_soc['ini'] + tmp_soc['delta']

		# Volcamos la salida al objeto
		self.soc = tmp_soc

	def get_voltages(self):
    	################################################################################################################################# FALTA HACER EL CALCULO DE CADA MODULO INDIVIDUAL RESPECTO A LOS STATS #################################################################################################################################

		model = self.model

		# Buscamos dónde empiezan las tensiones
		voltages_point = self.init_point

		# Primero buscamos avanzando en el texto
		n_lines_limit = 100 # Maximum amount of lines to move into searching for Voltages

		cont = True
		above = False
		found = False

		# Buscamos hacia abajo
		if not above:
			while cont:
				# Buscamos que no haya timestamp (cuando se muestran las tensiones de celda no hay timestamp) y que info no sea vacío
				if (model.df_info['time'][voltages_point] == model.time_emulator) and (not model.df_info['info'][voltages_point] == ''):
					cont = False
					above = False
					found = True
				else:
					# Si no encontramos un SoC, subimos una línea (pueden haberse escrito datos en medio del update...)
					voltages_point += 1

				# Si nos alejamos X puntos hacia abajo respecto del inicio o llegamos al final, paramos y buscamos en la otra dirección
				if (abs(voltages_point - self.init_point) > n_lines_limit) or (voltages_point >= len(model.df_info)):
					above = True
					cont = False
					found = False
		# Buscamos hacia arriba
		if (above) and (not found):
			# Si avanzando no encontramos, buscmaos retrocediendo en el texto
			voltages_point = self.init_point

			cont = True
			above = True
			found = False
			while cont:
				# Buscamos que no haya timestamp (cuando se muestran las tensiones de celda no hay timestamp) y que info no sea vacío
				if (model.df_info['time'][voltages_point] == model.time_emulator) and (not model.df_info['info'][voltages_point] == ''):
					voltages_point -= (model.n_modules - 1)
					cont = False
					above = False
					found = True
				else:
					voltages_point -= 1

				# Si nos alejamos n_lines_limit puntos hacia arriba respecto del inicio o llegamos al principio, paramos y consideramos que no existe la información
				if (abs(voltages_point - self.init_point) > n_lines_limit) or (voltages_point == 0):
					above = False
					cont = False
					found = False

		# Info sobre las teniones en el update
		tmp_voltages = []
		if found:
			# Si hemos encontrado la información, la extraemos segun el formato
			for i in range(model.n_modules):
				tmp_mec = []

				try:
				# if model.update_format == 0:
					tmp_name = model.df_info['info'][voltages_point + i].split(',')[0].split('_')[0]
					tmp_min = model.df_info['info'][voltages_point + i].split(',')[0].split('=')[-1].split(' ')[0]
					tmp_min = int(tmp_min)
					tmp_max = model.df_info['info'][voltages_point + i].split(',')[1].split('=')[-1].split(' ')[0]
					tmp_max = int(tmp_max)
				except:
				# elif model.update_format == 1:
					tmp_name = model.df_info['info'][voltages_point + i].split('=')[0].split('_')[0]
					tmp_max = model.df_info['info'][voltages_point + i].split('=')[1].split('/')[0]
					tmp_max = int(tmp_max)
					tmp_min = model.df_info['info'][voltages_point + i].split('=')[1].split('/')[1]
					tmp_min = int(tmp_min)

				tmp_mec.append(tmp_name)
				tmp_mec.append(self.datetime)
				tmp_mec.append(tmp_min)
				tmp_mec.append(tmp_max)

				tmp_voltages.append(tmp_mec)

			tmp_df_voltages = pd.DataFrame(tmp_voltages, columns = ['mec', 'datetime', 'vmin', 'vmax'])
		else:
    	# Si no hemos encontrado la información, provocamos una excepción
			raise ValueError("Update voltages not found")

		# Volcamos la salida al objeto
		self.df_voltages = tmp_df_voltages

	def get_update_type(self):
		if self.soc['delta'] >= 0:
			type = 'up'
		else:
			type = 'down'
		
		self.type = type

	def get_updater_mec(self):
		if self.type == 'up':
			updater_mec = self.df_voltages[self.df_voltages['vmax'] == self.df_voltages['vmax'].max()]

			if (len(updater_mec) > 1):
    			# En el caso de que haya más de un módulo que han actualizado con la máxima, nos quedamos la máxima de las mínimas
				updater_mec = updater_mec[updater_mec['vmin'] == updater_mec['vmin'].max()]

				if (len(updater_mec) > 1):
    				# En el caso de que siga habiendo más de un módulo que han actualizado con la máxima, cogemos el primero de ellos
					updater_mec = updater_mec.iloc[[0]]
		elif self.type == 'down':
			updater_mec = self.df_voltages[self.df_voltages['vmin'] == self.df_voltages['vmin'].min()]

			if (len(updater_mec) > 1):
    			# En el caso de que haya más de un módulo que han actualizado con la mínima, nos quedamos la mínima de las máximas
				updater_mec = updater_mec[updater_mec['vmax'] == updater_mec['vmax'].min()]

				if (len(updater_mec) > 1):
    				# En el caso de que siga habiendo más de un módulo que han actualizado con la mínima, cogemos el primero de ellos
					updater_mec = updater_mec.iloc[[0]]

		updater = str(updater_mec['mec'].values[0])
		value = int(updater_mec['vmax'].values)

		ret_val = []
		ret_val.append(updater)
		ret_val.append(value)

		return ret_val		

	def get_stats(self):
		ret_val = {}
		ret_val['max'] = {}
		ret_val['min'] = {}

		#### Stats Vmax
		# mea
		ret_val['max']['mea'] = round(self.df_voltages['vmax'].mean(), 2)
		# std
		ret_val['max']['std'] = round(self.df_voltages['vmax'].std(), 2)
		# disp = std/mea
		try:
			ret_val['max']['disp']  = round(ret_val['max']['std'] / ret_val['max']['mea'], 4)
		except:
			ret_val['max']['disp'] = 0
		# max
		ret_val['max']['max'] = self.df_voltages['vmax'].max()
		# max rel = max/mea
		try:
			ret_val['max']['max_rel']  = round(abs(1 - (ret_val['max']['max'] / ret_val['max']['mea'])), 4)
		except:
			ret_val['max']['max_rel'] = 0
		# min
		ret_val['max']['min'] = self.df_voltages['vmax'].min()
		# min rel = min/mea
		try:
			ret_val['max']['min_rel']  = round(abs(1 - (ret_val['max']['min'] / ret_val['max']['mea'])), 4)
		except:
			ret_val['max']['min_rel'] = 0			


		#### Stats Vmin
		# mea
		ret_val['min']['mea'] = round(self.df_voltages['vmin'].mean(), 2)
		# std
		ret_val['min']['std'] = round(self.df_voltages['vmin'].std(), 2)
		# disp = std/mea
		try:
			ret_val['min']['disp']  = round(ret_val['min']['std'] / ret_val['min']['mea'], 4)
		except:
			ret_val['min']['disp'] = 0		
		# max
		ret_val['min']['max'] = self.df_voltages['vmin'].max()
		# max rel = max/mea
		try:
			ret_val['min']['max_rel']  = round(abs(1 - (ret_val['min']['max'] / ret_val['min']['mea'])), 4)
		except:
			ret_val['min']['max_rel'] = 0		
		# min
		ret_val['min']['min'] = self.df_voltages['vmin'].min()
		# min rel = min/mea
		try:
			ret_val['min']['min_rel']  = round(abs(1 - (ret_val['min']['min'] / ret_val['min']['mea'])), 4)
		except:
			ret_val['min']['min_rel'] = 0	

		return ret_val

	def print_update(self, verbose = True):
		message = "\n***************************************************************************************************"
		message += f"\n\n{self.type.upper()} Update (line {self.init_point}): SoC from {self.soc['ini']}% to {self.soc['end']}% (delta {self.soc['delta']}%)"
		message += " \n\nVoltages:"
		message += self.df_voltages.to_string(header = True, col_space = 10)

		stats = self.get_stats()
		message += "\n\nStats MAX =>\t"
		message += f"  Mea: {stats['max']['mea']}\t Std: {stats['max']['std']} ({round(100*stats['max']['disp'], 2)}%)\t Max: {stats['max']['max']} ({round(100*stats['max']['max_rel'], 2)}%)\t Min: {stats['max']['min']} ({round(100*stats['max']['min_rel'], 2)}%)"

		message += "\nStats MIN =>\t"
		message += f"  Mea: {stats['min']['mea']}\t Std: {stats['min']['std']} ({round(100*stats['min']['disp'], 2)}%)\t Max: {stats['min']['max']} ({round(100*stats['min']['max_rel'], 2)}%)\t Min: {stats['min']['min']} ({round(100*stats['min']['min_rel'], 2)}%)"

		updater_mec = self.get_updater_mec()
		message += "\n\nUpdater MEC =>\t"
		if self.type == 'up':
			message += f"  {updater_mec[0]} -> Vcell MAX = {updater_mec[1]} mV"
		elif self.type == 'down':
			message += f"  {updater_mec[0]} -> Vcell MIN = {updater_mec[1]} mV"

		message += "\n\n***************************************************************************************************"
		self.model.tmp_updates_log.append(message)

##################################### MAIN #####################################
if __name__ == '__main__':
	# file_name = 'console_out.txt'
	# file_name = 'console_out_20210908-123208.txt'
	# file = getcwd() + '\\' + file_name

	Diagnosis_Tool()