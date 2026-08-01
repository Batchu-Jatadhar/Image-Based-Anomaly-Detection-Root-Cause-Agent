import { Module } from '@nitrostack/core';
import { InspectionTools } from './inspection.tools.js';
import { InspectionResources } from './inspection.resources.js';
import { InspectionPrompts } from './inspection.prompts.js';

@Module({
  name: 'inspection',
  description: 'Smart Manufacturing Defect Detection & Root Cause Analysis Module',
  providers: [
    InspectionTools,
    InspectionResources,
    InspectionPrompts
  ]
})
export class InspectionModule {}
