import {
  Entity, PrimaryGeneratedColumn, Column, ManyToOne, JoinColumn,
  CreateDateColumn, UpdateDateColumn
} from "typeorm";
import { Season } from "./Season";

@Entity("car")
export class Car {
  @PrimaryGeneratedColumn("uuid")
  id: string;

  @ManyToOne(() => Season)
  @JoinColumn()
  season: Season;

  @Column()
  model_name: string;

  @Column("jsonb", { nullable: true })
  spec_details: any;

  @CreateDateColumn()
  created_at: Date;

  @UpdateDateColumn()
  updated_at: Date;
}
