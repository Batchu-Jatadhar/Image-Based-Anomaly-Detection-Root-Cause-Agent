import { PromptDecorator as Prompt, ExecutionContext } from '@nitrostack/core';

export class InspectionPrompts {
  @Prompt({
    name: 'inspect_component_help',
    description: 'Guided assistance for selecting component categories, running anomaly detection, and generating inspection reports.',
    arguments: [
      {
        name: 'component_type',
        description: 'Type of manufacturing component (e.g. bottle, steel_sheet, metal_part)',
        required: false
      }
    ]
  })
  async getHelpPrompt(args: any, ctx: ExecutionContext) {
    const compType = args?.component_type || 'industrial component';
    return [
      {
        role: 'user' as const,
        content: `Please guide me through inspecting a ${compType}. What parameters are required for running the anomaly detection, localization, and root-cause analysis pipeline?`
      }
    ];
  }
}
