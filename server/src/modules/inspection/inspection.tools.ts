import { ToolDecorator as Tool, ExecutionContext, z } from '@nitrostack/core';
import { exec } from 'child_process';
import * as path from 'path';
import { promisify } from 'util';

const execAsync = promisify(exec);

export class InspectionTools {
  @Tool({
    name: 'inspect_industrial_image',
    description: 'Runs end-to-end AI defect detection, localization, RAG root cause analysis, and report verification on an industrial image.',
    inputSchema: z.object({
      image_path: z.string().describe('Path to the industrial component image to inspect'),
      dataset: z.enum(['mvtec', 'neu', 'severstal']).optional().default('mvtec').describe('Target dataset paradigm: mvtec (anomaly detection), neu (classification), or severstal (segmentation)'),
      category: z.string().optional().default('bottle').describe('MVTec category name (e.g. bottle, cable, tile, transistor)')
    }),
    examples: {
      request: {
        image_path: 'dataset/bottle/test/broken_large/000.png',
        dataset: 'mvtec',
        category: 'bottle'
      },
      response: {
        success: true,
        vision_output: {
          label: 'anomaly',
          confidence: 0.9998,
          bbox: [0.25, 0.25, 0.50, 0.50],
          heatmap_overlay_path: 'outputs/heatmaps/pred_overlay.png'
        },
        report: {
          impression: 'High-severity structural anomaly detected on component surface.',
          root_cause: 'Material fatigue from prolonged cyclic loading.',
          supporting_evidence: ['Visual presence of 3cm surface crack'],
          recommended_next_steps: ['De-energize assembly and replace component within 48 hours']
        },
        verification: {
          confidence_score: 0.95,
          verified: true
        }
      }
    }
  })
  async inspectImage(input: any, ctx: ExecutionContext) {
    ctx.logger.info('Executing industrial image inspection pipeline', {
      image_path: input.image_path,
      dataset: input.dataset,
      category: input.category
    });

    const repoRoot = path.resolve(__dirname, '../../../../');
    const pythonScript = `
import json, sys
sys.path.insert(0, r"${repoRoot}")
from src.agents.orchestrator import run_pipeline
meta = {"machine_id": "M_NITRO_01", "operator": "NitroStudio"}
res = run_pipeline(r"${input.image_path}", meta)
print(json.dumps(res))
`;

    try {
      const { stdout, stderr } = await execAsync(`python -c "${pythonScript.replace(/\n/g, ' ')}"`, {
        cwd: repoRoot,
        env: { ...process.env, PYTHONPATH: repoRoot }
      });

      const jsonStart = stdout.indexOf('{');
      const jsonEnd = stdout.lastIndexOf('}');
      if (jsonStart !== -1 && jsonEnd !== -1) {
        const cleanJson = stdout.substring(jsonStart, jsonEnd + 1);
        const parsed = JSON.parse(cleanJson);
        return {
          success: true,
          ...parsed
        };
      }

      return {
        success: true,
        raw_output: stdout
      };
    } catch (err: any) {
      ctx.logger.error('Failed to execute inspection pipeline', { error: err.message });
      return {
        success: false,
        error: err.message
      };
    }
  }

  @Tool({
    name: 'get_supported_vision_categories',
    description: 'Retrieves all supported industrial datasets, model backbones, and defect category taxonomies.',
    inputSchema: z.object({}),
    examples: {
      request: {},
      response: {
        mvtec_categories: ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper'],
        neu_classes: ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches'],
        severstal_classes: ['Class 1', 'Class 2', 'Class 3', 'Class 4']
      }
    }
  })
  async getCategories(input: any, ctx: ExecutionContext) {
    return {
      mvtec_categories: [
        'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
      ],
      neu_classes: ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches'],
      severstal_classes: ['Class 1', 'Class 2', 'Class 3', 'Class 4']
    };
  }
}
