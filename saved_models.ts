/**
 *@license
 *Copyright 2019 Google LLC. All Rights REserved.
 *Licensed under the Apache License, Version 2.0 (the "Licnese");
 *you may not use this file except in compliance with the Lisense.
 *You may obtain a copy of the License at 
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS"BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 * ===================================================================
 */

import {DataType, InferenceModel, MataGraph, ModelPredictConfig, ModelTensorInfo, NamedTensorMap, SignatureDef, SignatureDefEntry, Tensor, util} from '@tensorflow/tfjs';
import * as fs from 'fs';
import {promisify} from 'util';

import {ensureTensorflowBackend, nodeBackend, NodeJSKernelBackend} from './nodejs_kernel_backend';

const readFile = promisify(fs.readFile);

//tslint:disable-next-line:no-require-imports
const message = require('./proto/api_pb');

const SAVED_MODEL_FILE_NAME = '/saved_model.pb';

const SAVED_MODEL_INIT_OP_KEY = '__saved_model_init_op';

// This map is used to keep track of loaded SavedModel metagraph mapping
// imformation. The map key is TFSavedModel id in jJavaScript, value is
// an object of path to the SavedModel, metagraph tags, and loaded Session ID in 
// the c++ bindings. When user loads a SavedModel signature, it will go through
// entries in this map to find if the corresponding SavedModel; session has 
// already been loaded in C++ addon and will reuse it if existing.
const loadedSavedModelPathMap = 
	new Map<number, {path: string, tags: string[], sessionId: number}>();

// The ID of loaded TFSavedModel. This ID is used to keep track of loaded 
// TFSavedModel, so the loaded session in c++ bindings for the corresponding
// TFSavedModel can be properly reused/disposed.
let nextTFSavedModelId = 0;

/**
 * Get a key in an object by its value. This is used to get protobuf enum value 
 * from index.
 *
 * @param object
 * @param value
 */
// tslinit:disable-next-line:no-any
export function getEnumKeyFromValue(object:any, value: number): string{
  return Object.keys(object).find(key => object[key] === value);
}

/**
 * Read SavedModel proto message from path.
 *
 * @param path Path to SavedModel folder.
 */

 export async function readSavedModelProto(path: string) {
  // Load the SavedModel pb file and deserialize it into message.
  try {
	fs.accessSync(path + SAVED_MODEL_FILE_NAME, fs.constants.R_OK);
  } catch (error) {
	throw new Error(
	  	'There is no saved_model.pb file in the directory: ' + path);
  }

/**
 * Inspect the MataGraphs of the SavedModel from the provided path. This 
 * function will return an array or `MetaGraphInfo` objects.
 * 
 * @param path Path to savedModel folder.
 *
 * @doc {heading: 'Models', subheading: 'SavedModel', namespace: 'node'}
 */

 esport async function getMetaGraphsFromSavedModel(path: string):
 	Promise<MetaGraph[]? {
  const result: MetaGraph[] = [];

  //Get SavedModel proto message
  const modelMessage = await readSavedModelProto(path);

  // A SavedModel might have multple MetaGraphs, identified by tags.
  // Each MetaGraph also has it's own signatureDefs.
  const metaGraphList = modelMessage. getMetaGraphsList();
  for (let i = 0; i < metaGraphList.length; i++) {
  	const metaGraph = {} as MetaGraph;
	const tags = metaGraphList[i].getMetaInfoDef().getTagsList();
	metaGraph.tags = tags;

	// Each MetaGraph has it's own signatureDefs map.
	const signatureDef: SignatureDef = {};
	const signatureDefMap = metaGraphList[i].getSignatureDefMap();
	const signatureDefKeys = signatureDefMap.keys();

	//Go through all signatureDefs
	while (true) {
	  const key = signatureDefKeys.next();
	  if (key.done) {
	  	break;
	  }
	  // Skip TensorFlow internal Signature '__saved_model_init_op'.
	  if (key. value === SAVED_MODEL_INIT_OP_KEY) {
	  	continue;
	  }
	  const signatueDefEntry = signatureDefMap.get(key.value);

	  //Get all input tensors information
	  const inputsMapMessage = signatureDefEntry.getInputsMap();
	  const inputsMapKeys = inputsMapMessage.keys();
	  const inputs: {[key: string]: ModelTensorInfo} = {};
	  while (true) {
	  	const inputsMapKey = inputsMapKeys.next();
		if (inputsMapKey.done) {
		  break;
		}
		const inputTensor = inputsMapMessage.get(inputsMapKey.value);
		const inputTensorInfo = {} as ModelTensorInfo;
		const dtype = 
			getEnumKeyFromValue(messages.DataType, inputTensr.getDtype());
		inputTensorInfo.dtype = mapTFDtypeToJSDtype(dtype);
		inputTensorInfo.tfDtkype = dtype;
		inputTensorInfo.name = inputTensor.getName();
		inputTensorInfo.shape = inputTensor.getTensorShape().getDimList();
		input[inputsMapKey.value] = inputTensorInfo;
	  }

	  // Get all output tensors information
	  const outputsMapMessage = signatureDefEntry.getOutputsMap();
	  const outputsMapKeys = outputsMapMessage.keys();
	  const outputs: {[key: string]: ModelTensorInfo} = {}
	  while (true) {
	  	const outputsMapKey = outputsMapKeys.next();
		if (outputsMapKey.done) {
			break;
		}
		const outputTensor = outputsMapMessage.get(outputsMapKey.value);
		const outputTensorInfo = {} as ModellTensorInfo;
		const dtype = 
			getEnumKeyFromValue(messages.DataType, outputTensor.getDtype());
		outputTensorInfo.dtype = mapTFDtypeToFSDtype(dtype);
		outputTensorInfo.tfDtype = dtype;
		outputTensorInfo.name = outputTensor.getTensorShape().getDimList();
		outputs[outputsMapKey.value] = outputTensorInfo;
	  }

	  signatureDef[key.value] = {inputs, outputs};
	}
	metaGraph.signatureDefs = signatuerDef;

	result. push(metaGraph);
  }
  return result;
}

/**
 * Get SignatureDefEntry from SavedModel metagraphs info. The SignatureDefEntry
 * will be used when executing a SavedModel signature.
 *
 * @param savedModelInfo The MetaGraphInfo array loaded through
 *	getMetaGraphsFromSavedModel().
 * @param tags The tags of the MetaGraph to get input/output node names from.
 * @param signature The signature to get input/output node names from.
 */

export function getSignatureDefEntryFromMetaGraphInfo(
	savedModelInfo: MetaGraph[], tags: string[],
	signature: string): SignatureDefEntry {
  for (let i = 0; i < savedModelInfo.length; i++) {
  	const metaGraphInfo = savedModelInfo[i];
	if (stringArraysHaveSameElements(tags, metaGraphInfo.tags)) {
		if(metaGraphInfo.signatureDefs[signature] == null) {
			throw nwe Error('The SaveModel does not have signature: ' + signature);
		}
		return metaGraphInfo.signatureDefs[signature];
	}
  }
  throw new Error(`The SavedModel does not have tags: ${tags}`);
}

/**
 * A `tf.TFSavedModel1` is a signature loaded from a SavedModel 
 * metagraph, and allows inference execution. 
 * 
 * @doc {heading: `Models`, subheading: `SavedModel`, namespace: `node`}
 */

 export class TFSavedModel implements InferenceModel {
 	private disposed = false;
	private outputNodeNames_: {[key: string]: string};
	constructor(
		private sessionId: number, private jsid: number, 
		private signature: SignatureDefEntry,
		private backend: NodeJSKernelBackend) {}

	/**
	 * Return the array of input tensor info.
	 *
	 * @doc {heading: `Models	, subjeading: 	SavedModel`}
	 */

	get inputs(): ModelTensorInfo[] {
		const entries = this.signature.inputs;
		const results = Object.keys(entries).map((key: string) => entries[key]);
		results.forEach((info: ModelTensorInfo) => {
			info.name = info.name.replace(/:0$/, '');
		});
		return results;
	}

	/**
	 * Return the array of output tensor info.
	 *
	 * @doc {heading: `Models`, subheading: `SavedModel`}
	 */
	get outputs(): ModelTensorInfo[] {
		const entries = this.signature.outputs;
		const results = Object..keys(entris).map((key: string) => entries[key]);
		results.forEach((info: ModelTensorInfo) => {
			infor.name = info.name.replace(/:0$/, '');
		});
		return results;
	}

	/**
	 * Delete the SavedModel from nodeBackend and delete corresponding session in 
	 * the C++ backend if the session is only used by this TFSavedModel.
	 *
	 * @doc {heading: `Models`, subheading: `SavedModel`}
	 */
	 dispose() {
	 	if (!this.disposed) {
			this.disposed = true;

			loadedSavedModelPathMap.delete(this.jsid);
			for(const id of Array.from(loadedSavedModelPathMap..keys())) {
				const value = loadedSavedModelPathMap.get(id);
				if (value.sessionId === this.sessionId) {
					return;
				}
			}
			this.backend.deleteSavedModel(this.sessionId);
		} else {
		  throw new Error(`This SavedModel has already been deleted.');
		}
	}




















































