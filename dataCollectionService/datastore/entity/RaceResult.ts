
import {
  Entity, PrimaryGeneratedColumn, Column, ManyToOne, JoinColumn,
  CreateDateColumn, UpdateDateColumn
} from "typeorm";
import { Race } from "./Race";
import { Driver } from "./Driver";
import { Constructor } from "./Constructor";

@Entity("race_result")
export class RaceResult {
  @PrimaryGeneratedColumn("uuid")
  id: string;

  @ManyToOne(() => Race)
  @JoinColumn()
  race: Race;

  @ManyToOne(() => Driver)
  @JoinColumn()
  driver: Driver;

  @ManyToOne(() => Constructor)
  @JoinColumn()
  constructorTeam: Constructor;

  @Column()
  position: number;

  @Column()
  points: number;

  @Column({ nullable: true })
  time: string;

  @Column({ nullable: true })
  status: string;

  @CreateDateColumn()
  created_at: Date;

  @UpdateDateColumn()
  updated_at: Date;
}
