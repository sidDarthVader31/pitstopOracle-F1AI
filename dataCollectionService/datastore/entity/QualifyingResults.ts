
import {
  Entity, PrimaryGeneratedColumn, Column, ManyToOne, JoinColumn,
  CreateDateColumn, UpdateDateColumn
} from "typeorm";
import { Race } from "./Race";
import { Driver } from "./Driver";

@Entity("qualifying_result")
export class QualifyingResult {
  @PrimaryGeneratedColumn("uuid")
  id: string;

  @ManyToOne(() => Race)
  @JoinColumn()
  race: Race;

  @ManyToOne(() => Driver)
  @JoinColumn()
  driver: Driver;

  @Column()
  position: number;

  @Column({ nullable: true })
  q1_time: string;

  @Column({ nullable: true })
  q2_time: string;

  @Column({ nullable: true })
  q3_time: string;

  @CreateDateColumn()
  created_at: Date;

  @UpdateDateColumn()
  updated_at: Date;
}
